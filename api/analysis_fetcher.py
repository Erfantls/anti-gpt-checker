import hashlib
import json
from typing import Optional, Tuple, List

from fastapi import Depends, APIRouter, HTTPException, status, Query
from fastapi.responses import FileResponse, JSONResponse
from starlette import status

from api.api_models.document import DocumentInDB, DocumentStatus
from api.db_calls import router, dao_document, dao_analysis, dao_attribute
from api.server_config import API_ATTRIBUTES_COLLECTION_NAME, API_DEBUG, API_MONGODB_DB_NAME, API_HISTOGRAMS_PATH, \
    API_DEBUG_USER_ID, API_MOST_IMPORTANT_ATTRIBUTES
from api.server_dao.analysis import DAOAsyncAnalysis
from api.server_dao.document import DAOAsyncDocument
from api.api_models.analysis import AnalysisInDB, AnalysisData, AnalysisType, AnalysisStatus
from api.api_models.request import LightbulbScoreRequestData
from api.api_models.response import BackgroundTaskStatusResponse, AnalysisResultsResponse, \
    LightbulbScoreResponse, DocumentPreprocessingStillRunningResponse, DocumentPreprocessingFinishedResponse, \
    HistogramDataDTO, DocumentDataWithAnalyses, DocumentLevelAnalysis, ChunkLevelAnalysis, ChunkLevelSubanalysis, \
    UserDocumentsWithAnalyses, DocumentLevelAnalysisAdditionalDetails, ChunkLevelAnalysisAdditionalDetails, \
    ChunkLevelSubanalysisAdditionalDetails, DocumentWithAnalysesAdditionalDetails
from api.analyser import compare_2_hists, compute_histogram_data
from api.security import verify_token
from api.utils import _validate_analysis, _handle_analysis_status, calculate_lightbulb_scores

from dao.attribute import DAOAsyncAttributePL
from models.attribute import AttributePLInDB

router = APIRouter()
dao_analysis: DAOAsyncAnalysis = DAOAsyncAnalysis()
dao_document: DAOAsyncDocument = DAOAsyncDocument()
dao_attribute: DAOAsyncAttributePL = DAOAsyncAttributePL(collection_name=API_ATTRIBUTES_COLLECTION_NAME,
                                                         db_name=API_MONGODB_DB_NAME)


@router.get("/document-preprocessing-status",
            response_model=DocumentPreprocessingStillRunningResponse | DocumentPreprocessingFinishedResponse,
            status_code=status.HTTP_200_OK)
async def document_preprocessing_status(document_hash: str,
                                        user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    document: Optional[DocumentInDB] = await dao_document.find_one_by_query(
        {'document_hash': document_hash, 'owner_id': user_id})
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document with the specified hash does not exist"
        )

    if document.document_status == DocumentStatus.PREPROCESS_RUNNING:
        return DocumentPreprocessingStillRunningResponse()
    elif document.document_status == DocumentStatus.READY_FOR_ANALYSIS:
        return DocumentPreprocessingFinishedResponse()
    else:
        raise


@router.get("/document-analysis-status",
            response_model=BackgroundTaskStatusResponse,
            status_code=status.HTTP_200_OK)
async def document_analysis_status(analysis_id: str,
                                   _: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    analysis: Optional[AnalysisInDB] = await dao_analysis.find_one_by_query({'analysis_id': analysis_id})
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analysis with the specified hash does not exist"
        )

    return _handle_analysis_status(analysis)


@router.get("/document-analysis-result",
            response_model=AnalysisResultsResponse | BackgroundTaskStatusResponse,
            status_code=status.HTTP_200_OK)
async def document_analysis_results(analysis_id: str,
                                    _: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    validation_result = await _validate_analysis(analysis_id)
    if isinstance(validation_result, Tuple):
        analysis, attribute = validation_result
    else:
        return validation_result

    analysis_data: AnalysisData = AnalysisData.from_analysis_and_attribute(analysis, attribute)

    return AnalysisResultsResponse(
        analysis_data=analysis_data
    )


# this is closer to get endpoint, however, it is a post endpoint because it requires a body with attribute names
@router.post("/lightbulbs-scores",
             response_model=LightbulbScoreResponse,
             status_code=status.HTTP_200_OK)
async def lightbulb_score(analysis_id: str, request_data: LightbulbScoreRequestData,
                          _: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    validation_result = await _validate_analysis(analysis_id)
    if isinstance(validation_result, Tuple):
        analysis, attribute = validation_result
    else:
        return validation_result
    attribute_names = request_data.attribute_names
    lightbulb_score_data = await calculate_lightbulb_scores(attribute, attribute_names)

    return LightbulbScoreResponse(
        lightbulb_scores=lightbulb_score_data
    )


@router.get("/histogram-image",
            status_code=status.HTTP_200_OK)
async def get_graph_image(analysis_id: str, attribute_name: str,
                          _: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    validation_result = await _validate_analysis(analysis_id)
    if isinstance(validation_result, Tuple):
        analysis, attribute = validation_result
    else:
        return validation_result

    attribute_dict = attribute.to_flat_dict_normalized()
    if attribute_name not in attribute_dict:
        raise HTTPException(
            status_code=404,
            detail="No attributes found connected with the analysis"
        )

    compare_2_hists(attribute_name=attribute_name, file_name=f"{analysis_id}_{attribute_name}",
                    additional_value=attribute_dict[attribute_name])
    image_path = f"{API_HISTOGRAMS_PATH}/{analysis_id}_{attribute_name}.png"
    return FileResponse(image_path, media_type="image/png")

@router.get("/histogram-data", response_model=HistogramDataDTO, status_code=status.HTTP_200_OK)
async def get_graph_summary(
    analysis_id: str,
    attribute_name: str,
    num_bins: int = Query(21, gt=1, le=100),
    min_value: Optional[float] = Query(None),
    max_value: Optional[float] = Query(None),
    existing_hash: Optional[str] = Query(None, alias="hash"),
    _: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID
):
    validation_result = await _validate_analysis(analysis_id)
    if isinstance(validation_result, Tuple):
        analysis, attribute = validation_result
    else:
        return validation_result

    attribute_dict = attribute.to_flat_dict_normalized()
    if attribute_name not in attribute_dict:
        raise HTTPException(
            status_code=404,
            detail="No attributes found connected with the analysis"
        )

    dto = compute_histogram_data(
        attribute_name=attribute_name,
        num_bin=num_bins,
        min_value=min_value,
        max_value=max_value,
        additional_value=attribute_dict[attribute_name]
    )

    if existing_hash and existing_hash == dto.object_hash:
        return JSONResponse(content={"detail": "Object hash did not change"}, status_code=200)

    return dto


@router.get("/user-document-with-analyses-overview",
            response_model=DocumentDataWithAnalyses,
            status_code=status.HTTP_200_OK)
async def get_document_with_analyses_overview(document_hash: str, user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    return await _get_document_with_analyses_overview(document_hash=document_hash, user_id=user_id)

@router.get("/user-documents-with-analyses-overview",
            response_model=UserDocumentsWithAnalyses,
            status_code=status.HTTP_200_OK)
async def get_documents_with_analyses_overview(start_index: int = 0, limit: Optional[int] = None, user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    document_hashes: List[str] = await dao_document.find_document_hash_by_query_paginated({'owner_id': user_id}, start_index, limit)
    documents_with_analyses: List[DocumentDataWithAnalyses] = []
    for document_hash in document_hashes:
        try:
            document_data_with_analyses = await _get_document_with_analyses_overview(document_hash=document_hash, user_id=user_id)
            documents_with_analyses.append(document_data_with_analyses)
        except HTTPException as e:
            if e.status_code == 404:
                continue

    # get hash of the documents with analyses
    json_str = json.dumps([obj.dict() for obj in documents_with_analyses], sort_keys=True)
    owned_data_hash = hashlib.sha256(json_str.encode()).hexdigest()

    return UserDocumentsWithAnalyses(
        documents_with_analyses=documents_with_analyses,
        owned_data_hash=owned_data_hash
    )


@router.get("/user-document-with-analyses-details",
            response_model=DocumentWithAnalysesAdditionalDetails,
            status_code=status.HTTP_200_OK)
async def get_user_document_with_analyses_details(document_hash: str, user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    document: DocumentInDB = await dao_document.find_one_by_query({'document_hash': document_hash, 'owner_id': user_id})
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document with the specified hash does not exist"
        )
    document_level_analysis_details = DocumentLevelAnalysisAdditionalDetails(analysed_text=document.plaintext_content)
    chunk_analyses: list[AnalysisInDB] = await dao_analysis.find_many_by_query(
        {'document_hash': document.document_hash, 'type': AnalysisType.CHUNK_LEVEL, 'status': AnalysisStatus.FINISHED})
    if len(chunk_analyses) == 0:
        chunk_level_analysis_details = ChunkLevelAnalysisAdditionalDetails(subanalyses_details=[])
    else:
        subanalyses_details = []
        attribute_id = chunk_analyses[0].attributes_id
        attribute: AttributePLInDB = await dao_attribute.find_by_id(attribute_id)
        if not attribute:
            raise HTTPException(
                status_code=404,
                detail="Attribute for the specified analysis does not exist"
            )
        for chunk_attributes in attribute.partial_attributes:
            subanalyses_details.append(ChunkLevelSubanalysisAdditionalDetails(
                identifier=chunk_attributes.index,
                analysed_text=chunk_attributes.partial_text
            ))
        chunk_level_analysis_details = ChunkLevelAnalysisAdditionalDetails(subanalyses_details=subanalyses_details)

    return DocumentWithAnalysesAdditionalDetails(
        document_hash=document.document_hash,
        document_level_analysis_details=document_level_analysis_details,
        chunk_level_analysis_details=chunk_level_analysis_details,
    )


async def _get_document_with_analyses_overview(document_hash: str, user_id: str):
    document: DocumentInDB = await dao_document.find_one_by_query({'document_hash': document_hash, 'owner_id': user_id})
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document with the specified hash does not exist"
        )

    analyses: list[AnalysisInDB] = await dao_analysis.find_many_by_query({'document_hash': document.document_hash, 'type': AnalysisType.DOCUMENT_LEVEL, 'status': AnalysisStatus.FINISHED})
    if len(analyses) == 0:
        return DocumentDataWithAnalyses(
            document_hash=document.document_hash,
            document_status=document.document_status,
            document_name=document.document_name,
            document_upload_date=document.created_at.isoformat(),
            document_level_analysis=None,
            chunk_level_analyses=None
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

    return DocumentDataWithAnalyses(
        document_hash=document.document_hash,
        document_status=document.document_status,
        document_name=document.document_name,
        document_upload_date=document.created_at.isoformat(),
        document_level_analysis=document_level_analysis,
        chunk_level_analyses=chunk_level_analysis
    )
