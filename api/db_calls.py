from typing import Optional

from fastapi import Depends, APIRouter, status, HTTPException

from api.server_config import API_ATTRIBUTES_COLLECTION_NAME, API_DEBUG, API_MONGODB_DB_NAME, \
    API_MOST_IMPORTANT_ATTRIBUTES, API_DEBUG_USER_ID

from api.api_models.document import DocumentInDB
from api.api_models.analysis import AnalysisInDB, AnalysisStatus, AnalysisData, AnalysisType
from api.api_models.response import DocumentsOfUserResponse, \
    AnalysesOfDocumentsResponse, DocumentsOfUserWithAnalysisResponse, AnalysisWithLightbulbs, DocumentWithAnalysis, \
    ChunkLevelSubanalysis, DocumentLevelAnalysis, ChunkLevelAnalysis, DocumentDataWithAnalyses

from api.security import verify_token

from api.server_dao.analysis import DAOAsyncAnalysis
from api.server_dao.document import DAOAsyncDocument
from api.server_dao.user import DAOAsyncUser
from api.utils import calculate_lightbulb_scores
from dao.attribute import DAOAsyncAttributePL
from models.attribute import AttributePLInDB

router = APIRouter()
dao_user: DAOAsyncUser = DAOAsyncUser()
dao_document: DAOAsyncDocument = DAOAsyncDocument()
dao_analysis: DAOAsyncAnalysis = DAOAsyncAnalysis()
dao_attribute: DAOAsyncAttributePL = DAOAsyncAttributePL(collection_name=API_ATTRIBUTES_COLLECTION_NAME,
                                                         db_name=API_MONGODB_DB_NAME)

@router.get("/get-documents-of-user",
            response_model=DocumentsOfUserResponse,
            status_code=status.HTTP_200_OK)
async def get_documents_of_user(user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    documents: list[DocumentInDB] = await dao_document.find_many_by_query({'owner_id': user_id})

    return DocumentsOfUserResponse(documents=documents)

@router.get("/get-analysed-documents-of-user",
            response_model=DocumentsOfUserWithAnalysisResponse,
            status_code=status.HTTP_200_OK)
async def get_analysed_documents_of_user(user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    documents: list[DocumentInDB] = await dao_document.find_many_by_query({'owner_id': user_id})
    documents_with_analyses: list[DocumentWithAnalysis] = []
    for document in documents:
        analyses_with_lightbulbs: list[AnalysisWithLightbulbs] = []
        analyses: list[AnalysisInDB] = await dao_analysis.find_many_by_query({'document_hash': document.document_hash})
        for analysis in analyses:
            if analysis.status != AnalysisStatus.FINISHED:
                continue

            attribute: AttributePLInDB = await dao_attribute.find_by_id(analysis.attributes_id)
            if not attribute:
                continue

            lightbulb_scores = await calculate_lightbulb_scores(attribute, API_MOST_IMPORTANT_ATTRIBUTES)
            analyses_with_lightbulbs.append(AnalysisWithLightbulbs(
                analysis=analysis,
                attribute_in_db=attribute,
                lightbulb_scores=lightbulb_scores
            ))

        documents_with_analyses.append(DocumentWithAnalysis(
            document=document,
            analyses_with_lightbulbs=analyses_with_lightbulbs
        ))

    return DocumentsOfUserWithAnalysisResponse(documents_with_analyses=documents_with_analyses)

@router.get("/user-document-with-analyses-details",
            response_model=DocumentDataWithAnalyses,
            status_code=status.HTTP_200_OK)
async def get_document_with_analyses_details(document_hash: str, user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    document: DocumentInDB = await dao_document.find_one_by_query({'document_hash': document_hash, 'owner_id': user_id})

    analyses: list[AnalysisInDB] = await dao_analysis.find_many_by_query({'document_hash': document.document_hash, 'type': AnalysisType.DOCUMENT_LEVEL, 'status': AnalysisStatus.FINISHED})
    if len(analyses) == 0:
        raise HTTPException(
            status_code=404,
            detail="No finished document-level analyses found for the specified document"
        )

    newest_analyses: AnalysisInDB = sorted(analyses, key=lambda x: x.start_time, reverse=True)[0]

    attribute: AttributePLInDB = await dao_attribute.find_by_id(newest_analyses.attributes_id)
    if not attribute:
        raise HTTPException(
            status_code=404,
            detail="Attribute for the specified analysis does not exist"
        )

    document_level_analysis = DocumentLevelAnalysis(
        status=AnalysisStatus.FINISHED, # it has to be finished as we are fetching only finished analyses
        lightbulb_features=await calculate_lightbulb_scores(attribute, API_MOST_IMPORTANT_ATTRIBUTES)
    )

    chunk_analyses: list[AnalysisInDB] = await dao_analysis.find_many_by_query({'document_hash': document.document_hash, 'type': AnalysisType.CHUNK_LEVEL, 'status': AnalysisStatus.FINISHED})
    if len(chunk_analyses) == 0:
        chunk_level_analysis = ChunkLevelAnalysis(
            status=AnalysisStatus.NOT_FINISHED,
            subanalyses=[]
        )
    else:
        chunk_level_subanalyses: list[ChunkLevelSubanalysis] = []
        for chunk_attributes in attribute.partial_attributes:
            identifier = chunk_attributes.index
            lightbulb_scores = await calculate_lightbulb_scores(chunk_attributes.attribute, API_MOST_IMPORTANT_ATTRIBUTES)
            chunk_level_subanalyses.append(ChunkLevelSubanalysis(
                identifier=identifier,
                lightbulb_features=lightbulb_scores
            ))
        chunk_level_analysis = ChunkLevelAnalysis(
            status=AnalysisStatus.FINISHED,
            subanalyses=chunk_level_subanalyses
        )

    document_data_with_analyses = DocumentDataWithAnalyses(
        document_hash=document.document_hash,
        document_status=document.document_status,
        document_name=document.document_name,
        document_upload_date=document.created_at.isoformat(),
        document_level_analysis=document_level_analysis,
        chunk_level_analyses=chunk_level_analysis
    )

    return document_data_with_analyses




@router.get("/get-document",
            response_model=DocumentInDB,
            status_code=status.HTTP_200_OK)
async def get_document(document_id: str, _: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    document: Optional[DocumentInDB] = await dao_document.find_by_id(document_id)
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document with the specified hash does not exist"
        )

    return document

@router.get("/get-document-by-hash",
            response_model=DocumentInDB,
            status_code=status.HTTP_200_OK)
async def get_document_by_hash(document_hash: str, user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    document: Optional[DocumentInDB] = await dao_document.find_one_by_query({'document_hash': document_hash, 'owner_id': user_id})
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document with the specified hash does not exist"
        )

    return document

@router.get("/document-exists",
            response_model=dict,
            status_code=status.HTTP_200_OK)
async def get_document_by_hash(document_hash: str, user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    document: Optional[DocumentInDB] = await dao_document.find_one_by_query({'document_hash': document_hash, 'owner_id': user_id})
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document with the specified hash does not exist"
        )

    return {"document_name": document.document_name, "uploaded_at": document.created_at}


@router.get("/get-analysis-of-document",
            response_model=AnalysesOfDocumentsResponse,
            status_code=status.HTTP_200_OK)
async def get_analyses_of_document(document_id: str, _: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    document: Optional[DocumentInDB] = await dao_document.find_by_id(document_id)
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document with the specified hash does not exist"
        )
    return await _get_analyses_of_document_by_hash(document.document_hash)

@router.get("/get-analysis-of-document-by-hash",
            response_model=AnalysesOfDocumentsResponse,
            status_code=status.HTTP_200_OK)
async def get_analyses_of_document_by_hash(document_hash: str, _: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    return await _get_analyses_of_document_by_hash(document_hash)

async def _get_analyses_of_document_by_hash(document_hash: str):
    analyses: list[AnalysisInDB] = await dao_analysis.find_many_by_query({'document_hash': document_hash})

    analyses_data = []
    for analysis in analyses:
        if analysis.type == AnalysisType.CHUNK_LEVEL:
            # Skip chunk-level analyses as their results are store in the document_level analysis
            continue
        if analysis.status != AnalysisStatus.FINISHED:
            analyses_data.append(
                AnalysisData(analysis_id=analysis.analysis_id, document_hash=analysis.document_hash, full_features=None))
            continue

        attribute: AttributePLInDB = await dao_attribute.find_by_id(analysis.attributes_id)
        if not attribute:
            analyses_data.append(
                AnalysisData(analysis_id=analysis.analysis_id, document_hash=analysis.document_hash, full_features=None))
            continue

        analyses_data.append(AnalysisData.from_analysis_and_attribute(analysis, attribute))

    return AnalysesOfDocumentsResponse(analyses=analyses_data)
