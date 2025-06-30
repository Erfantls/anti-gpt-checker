from typing import Optional, Tuple

from fastapi import Depends, APIRouter
from fastapi.responses import FileResponse

from api.server_config import API_ATTRIBUTES_COLLECTION_NAME, API_DEBUG, API_MONGODB_DB_NAME, API_HISTOGRAMS_PATH
from api.server_dao.analysis import DAOAsyncAnalysis
from api.server_dao.document import DAOAsyncDocument
from api.api_models.analysis import AnalysisInDB, AnalysisData
from api.api_models.request import LightbulbScoreRequestData
from api.api_models.response import BackgroundTaskStatusResponse, AnalysisResultsResponse, \
    LightbulbScoreResponse, NoAnalysisFoundResponse, NoAttributeFoundResponse, \
    LightbulbScoreData, LightbulbScoreType
from api.analyser import calculate_lightbulb_score, compare_2_hists
from api.security import verify_token
from api.utils import _validate_analysis, _handle_analysis_status

from dao.attribute import DAOAsyncAttributePL

router = APIRouter()
dao_analysis: DAOAsyncAnalysis = DAOAsyncAnalysis()
dao_document: DAOAsyncDocument = DAOAsyncDocument()
dao_attribute: DAOAsyncAttributePL = DAOAsyncAttributePL(collection_name=API_ATTRIBUTES_COLLECTION_NAME, db_name=API_MONGODB_DB_NAME)

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

    analysis_data: AnalysisData = AnalysisData.from_analysis_and_attribute(analysis, attribute)

    return AnalysisResultsResponse(
        analysis_data=analysis_data
    )

# this is closer to get endpoint, however, it is a post endpoint because it requires a body with attribute names
@router.post("/lightbulbs-scores",
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

    compare_2_hists(attribute_name=attribute_name, file_name=f"{analysis_id}_{attribute_name}",
                    additional_value=attribute_dict[attribute_name])
    image_path = f"{API_HISTOGRAMS_PATH}/{analysis_id}_{attribute_name}.png"
    return FileResponse(image_path, media_type="image/png")
