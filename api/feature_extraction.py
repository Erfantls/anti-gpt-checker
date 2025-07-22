import traceback
import hashlib
from datetime import datetime
from typing import Optional

from fastapi import BackgroundTasks, Depends, APIRouter, HTTPException, status

from analysis.attribute_retriving import perform_full_analysis
from analysis.nlp_transformations import preprocess_text
from api.server_config import API_ATTRIBUTES_COLLECTION_NAME, API_DOCUMENTS_COLLECTION_NAME, API_DEBUG, \
    API_MONGODB_DB_NAME, API_DEBUG_USER_ID
from api.server_dao.analysis import DAOAsyncAnalysis, DAOAnalysis
from api.server_dao.document import DAOAsyncDocument, DAODocument
from api.api_models.analysis import Analysis, AnalysisType, AnalysisStatus
from api.api_models.document import DocumentInDB, Document, DocumentStatus
from api.api_models.request import PreprocessedDocumentRequestData
from api.security import verify_token
from dao.attribute import DAOAttributePL
from models.attribute import AttributePL

router = APIRouter()

dao_async_analysis: DAOAsyncAnalysis = DAOAsyncAnalysis()
dao_async_document: DAOAsyncDocument = DAOAsyncDocument()


@router.post("/add-document",
             response_model=dict,
             status_code=status.HTTP_201_CREATED
             )
async def post_document(preprocessed_document: PreprocessedDocumentRequestData,
                        user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    # Check if the document already exists
    existing_doc: Optional[DocumentInDB] = await dao_async_document.find_one_by_query(
        {"document_hash": preprocessed_document.document_hash, "owner_id": user_id})
    if existing_doc:
        raise HTTPException(
            status_code=409,
            detail="Document with the specified hash already exists, please use a different ID"
        )
    else:
        document = Document(
            document_name=preprocessed_document.document_name,
            document_status=DocumentStatus.READY_FOR_ANALYSIS if preprocessed_document.preprocessed_content is not None else DocumentStatus.PREPROCESS_RUNNING,
            document_hash=preprocessed_document.document_hash,
            plaintext_content=preprocessed_document.preprocessed_content,
            filepath=preprocessed_document.filepath,
            owner_id=user_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await dao_async_document.insert_one(document)
    return {"message": f"Document with name {preprocessed_document.document_name} has been inserted"}


@router.patch("/update-document",
              response_model=dict,
              status_code=status.HTTP_200_OK
              )
async def update_document(preprocessed_document: PreprocessedDocumentRequestData,
                          user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    # Check if the document already exists
    existing_doc: Optional[DocumentInDB] = await dao_async_document.find_one_by_query(
        {"document_hash": preprocessed_document.document_hash, "owner_id": user_id})
    if not existing_doc:
        raise HTTPException(
            status_code=404,
            detail="Document with the specified hash does not exist"
        )
    else:
        set_fields = {}
        if existing_doc.plaintext_content:
            set_fields['document_status'] = DocumentStatus.READY_FOR_ANALYSIS
        else:
            set_fields['document_status'] = DocumentStatus.PREPROCESS_RUNNING

        for field in preprocessed_document.dict():
            if preprocessed_document.dict()[field] is not None:
                set_fields[field] = preprocessed_document.dict()[field]
                if field == 'preprocessed_content':
                    set_fields['document_status'] = DocumentStatus.READY_FOR_ANALYSIS

        set_fields['updated_at'] = datetime.now()

        await dao_async_document.update_one({"document_hash": preprocessed_document.document_hash, "owner_id": user_id},
                                            {'$set': set_fields})
    return {"message": f"Document with name {preprocessed_document.document_name} has been updated"}


@router.post("/trigger-analysis",
             response_model=dict,
             status_code=status.HTTP_202_ACCEPTED)
async def trigger_document_analysis(document_hash: str, background_tasks: BackgroundTasks,
                                    perform_full_analysis: bool = False,
                                    user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    existing_doc: Optional[DocumentInDB] = await dao_async_document.find_one_by_query(
        {"document_hash": document_hash, "owner_id": user_id})
    if not existing_doc:
        raise HTTPException(
            status_code=404,
            detail="Document with the specified hash does not exist"
        )
    # generate analysis_id
    analysis_id = hashlib.sha256(f"{document_hash}_{user_id}_{datetime.now().isoformat()}".encode()).hexdigest()
    analysis = Analysis(
        analysis_id=analysis_id,
        type=AnalysisType.FULL if perform_full_analysis else AnalysisType.PARTIAL,
        status=AnalysisStatus.RUNNING,
        document_hash=document_hash,
        estimated_wait_time=30,
        start_time=datetime.now()
    )
    await dao_async_analysis.insert_one(analysis)
    background_tasks.add_task(_perform_analysis, analysis_id, document_hash, user_id)
    return {"message": f"Analysis of document {document_hash} has been triggered",
            "analysis_id": str(analysis_id)}


dao_analysis: DAOAnalysis = DAOAnalysis()
dao_document: DAODocument = DAODocument()
dao_attribute: DAOAttributePL = DAOAttributePL(collection_name=API_ATTRIBUTES_COLLECTION_NAME,
                                               db_name=API_MONGODB_DB_NAME)


def _perform_analysis(analysis_id: str, document_hash, user_id: str):
    document: DocumentInDB = dao_document.find_one_by_query({'document_hash': document_hash, 'owner_id': user_id})
    try:
        text_to_analyse = preprocess_text(document.plaintext_content)
        analysis_result = perform_full_analysis(text_to_analyse, 'pl')
        attribute_to_insert = AttributePL(
            referenced_db_name=API_DOCUMENTS_COLLECTION_NAME,
            referenced_doc_id=document.id,
            language="pl",
            is_generated=None,
            is_personal=None,
            **analysis_result.dict()
        )
        attributes_id = dao_attribute.insert_one(attribute_to_insert)
        dao_analysis.update_one({'analysis_id': analysis_id},
                                {'$set': {'status': AnalysisStatus.FINISHED, "attributes_id": attributes_id}})
    except Exception as e:
        dao_analysis.update_one({'analysis_id': analysis_id}, {'$set':
                                                                   {'status': AnalysisStatus.FAILED,
                                                                    'error_message': traceback.format_exc()}})
