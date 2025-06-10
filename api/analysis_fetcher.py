from datetime import timedelta, datetime
from typing import Optional, Tuple
from fastapi import Depends, APIRouter
from fastapi.responses import FileResponse

from api.server_config import API_ATTRIBUTES_COLLECTION_NAME, API_DEBUG
from api.server_dao.analysis import DAOAsyncAnalysis
from api.server_dao.document import DAOAsyncDocument
from api.api_models.analysis import AnalysisInDB, AnalysisStatus, AnalysisData
from api.api_models.request import LightbulbScoreRequestData
from api.api_models.response import BackgroundTaskStatusResponse, BackgroundTaskRunningResponse, AnalysisResultsResponse, \
    LightbulbScoreResponse, NoAnalysisFoundResponse, BackgroundTaskFailedResponse, NoAttributeFoundResponse, \
    LightbulbScoreData, LightbulbScoreType
from api.analyser import calculate_lightbulb_score, compare_2_hists
from api.security import verify_token

from dao.attribute import DAOAsyncAttributePL
from models.attribute import AttributePLInDB

router = APIRouter()
dao_analysis: DAOAsyncAnalysis = DAOAsyncAnalysis()
dao_document: DAOAsyncDocument = DAOAsyncDocument()
dao_attribute: DAOAsyncAttributePL = DAOAsyncAttributePL(collection_name=API_ATTRIBUTES_COLLECTION_NAME)

@router.get("/document-analysis-status",
            response_model=BackgroundTaskStatusResponse | NoAnalysisFoundResponse)
async def document_analysis_results(analysis_id: str, _: bool = Depends(verify_token) if not API_DEBUG else True):
    analysis: Optional[AnalysisInDB] = await dao_analysis.find_one_by_query({'analysis_id': analysis_id})
    if not analysis:
        return NoAnalysisFoundResponse()

    return _handle_analysis_status(analysis)


@router.get("/document-analysis-result",
            response_model=AnalysisResultsResponse | BackgroundTaskStatusResponse | NoAnalysisFoundResponse)
async def document_analysis_results(analysis_id: str, _: bool = Depends(verify_token) if not API_DEBUG else True):
    validation_result = await _validate_analysis(analysis_id)
    if isinstance(validation_result, Tuple):
        analysis, attribute = validation_result
    else:
        return validation_result

    return AnalysisResultsResponse(
        analysis_data=AnalysisData(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_id,
            full_features=attribute
        )
    )


@router.get("/lightbulbs-scores",
            response_model=LightbulbScoreResponse)
async def lightbulb_score(analysis_id: str, request_data: LightbulbScoreRequestData, _: bool = Depends(verify_token) if not API_DEBUG else True):
    validation_result = await _validate_analysis(analysis_id)
    if isinstance(validation_result, Tuple):
        analysis, attribute = validation_result
    else:
        return validation_result

    attribute_dict = attribute.to_flat_dict_normalized()
    lightbulb_score_data = []
    for attribute_name in request_data.attribute_names:
        if attribute_name not in attribute_dict:
            continue

        attribute_value = attribute_dict[attribute_name]  # FIXME check what category should the score be in
        lightbulb_score_value = calculate_lightbulb_score(attribute_value, attribute_name,
                                                          category=LightbulbScoreType.BIDIRECTIONAL)
        lightbulb_score_data.append(LightbulbScoreData(
            attribute_name=attribute_name,
            type=LightbulbScoreType.BIDIRECTIONAL,
            score=lightbulb_score_value
        ))

    return LightbulbScoreResponse(
        lightbulb_scores=lightbulb_score_data
    )
    # return list of lightbulb scores


@router.get("/graph-image")
async def get_graph_image(analysis_id: str, attribute_name: str, _: bool = Depends(verify_token) if not API_DEBUG else True):
    validation_result = await _validate_analysis(analysis_id)
    if isinstance(validation_result, Tuple):
        analysis, attribute = validation_result
    else:
        return validation_result

    attribute_dict = attribute.to_flat_dict_normalized()
    if attribute_name not in attribute_dict:
        return NoAttributeFoundResponse()

    compare_2_hists(attribute_name=attribute_name, file_name=f"{analysis_id}_{attribute_name}",)
    image_path = f"histograms/{analysis_id}_{attribute_name}.png"
    return FileResponse(image_path, media_type="image/png")


def _handle_analysis_status(analysis: AnalysisInDB) -> BackgroundTaskStatusResponse:
    if analysis.status == AnalysisStatus.FAILED:
        return BackgroundTaskFailedResponse(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_id,
            estimated_wait_time=0
        )
    elif analysis.status == AnalysisStatus.RUNNING:
        estimated_end_time = analysis.start_time + timedelta(seconds=analysis.estimated_wait_time)
        remaining_time = (estimated_end_time - datetime.now()).total_seconds()
        if remaining_time < 10:
            # if remaining time is less than 10 seconds, set it to 30 seconds
            remaining_time = 30
        return BackgroundTaskRunningResponse(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_id,
            estimated_wait_time=remaining_time
        )
    elif analysis.status == AnalysisStatus.FINISHED:
        return BackgroundTaskRunningResponse(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_id,
            estimated_wait_time=0
        )
    else:
        raise Exception(f"Unknown analysis status: {analysis.status}")


async def _validate_analysis(
        analysis_id: str) -> Tuple[AnalysisInDB, AttributePLInDB] | NoAnalysisFoundResponse | BackgroundTaskStatusResponse | NoAttributeFoundResponse:
    analysis: Optional[AnalysisInDB] = await dao_analysis.find_one_by_query({'analysis_id': analysis_id})
    if not analysis:
        return NoAnalysisFoundResponse()

    if analysis.status != AnalysisStatus.FINISHED:
        return _handle_analysis_status(analysis)

    attribute: AttributePLInDB = await dao_attribute.find_one_by_query({'_id': analysis.features_id})
    if not attribute:
        return NoAttributeFoundResponse()

    return analysis, attribute

