import traceback
import hashlib
from datetime import datetime

from fastapi import BackgroundTasks, Depends, APIRouter

from analysis.attribute_retriving import perform_full_analysis
from analysis.nlp_transformations import preprocess_text
from api.server_config import API_ATTRIBUTES_COLLECTION_NAME, API_DOCUMENTS_COLLECTION_NAME, API_DEBUG
from api.server_dao.analysis import DAOAsyncAnalysis, DAOAnalysis
from api.server_dao.document import DAOAsyncDocument, DAODocument
from api.api_models.analysis import Analysis, AnalysisType, AnalysisStatus
from api.api_models.document import DocumentInDB, Document
from api.api_models.request import PreprocessedDocumentRequestData
from api.security import verify_token
from config import init_all_polish_models
from dao.attribute import DAOAttributePL
from models.attribute import AttributePL

router = APIRouter()

dao_async_analysis: DAOAsyncAnalysis = DAOAsyncAnalysis()
dao_async_document: DAOAsyncDocument = DAOAsyncDocument()

@router.post("/add-document")
async def post_preprocess_document(preprocessed_document: PreprocessedDocumentRequestData,
                                   _: bool = Depends(verify_token) if not API_DEBUG else True):
    document = Document(
        document_id=preprocessed_document.document_id,
        plaintext_content=preprocessed_document.preprocessed_content,
        filepath=preprocessed_document.filepath
    )
    await dao_async_document.insert_one(document)
    return {"message": f"Document has been inserted"}


@router.post("/trigger-analysis")
async def trigger_document_analysis(document_id: str, background_tasks: BackgroundTasks,
                                    perform_full_analysis: bool = False, _: bool = Depends(verify_token) if not API_DEBUG else True):
    # generate analysis_id
    analysis_id = hashlib.sha256(f"{document_id}_{datetime.now().isoformat()}".encode()).hexdigest()
    analysis = Analysis(
        analysis_id=analysis_id,
        type=AnalysisType.FULL if perform_full_analysis else AnalysisType.PARTIAL,
        status=AnalysisStatus.RUNNING,
        document_id=document_id,
        estimated_wait_time=30,
        start_time=datetime.now()
    )
    await dao_async_analysis.insert_one(analysis)
    background_tasks.add_task(_perform_analysis, analysis_id, document_id)
    return {"message": f"Analysis of document {document_id} has been triggered",
            "analysis_id": str(analysis_id)}


dao_analysis: DAOAnalysis = DAOAnalysis()
dao_document: DAODocument = DAODocument()
dao_attribute: DAOAttributePL = DAOAttributePL(collection_name=API_ATTRIBUTES_COLLECTION_NAME)

def _perform_analysis(analysis_id: str, document_id):
    document: DocumentInDB = dao_document.find_one_by_query({'document_id': document_id})
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
        dao_attribute.insert_one(attribute_to_insert)
        dao_analysis.update_one({'analysis_id': analysis_id}, {'$set': {'status': AnalysisStatus.FINISHED}})
    except Exception as e:
        dao_analysis.update_one({'analysis_id': analysis_id}, {'$set':
                                                                   {'status': AnalysisStatus.FAILED,
                                                                    'error_message': traceback.format_exc()}})


