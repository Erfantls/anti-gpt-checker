from datetime import timedelta, datetime
from typing import Tuple, Optional

from api.api_models.analysis import AnalysisInDB, AnalysisStatus
from api.api_models.response import NoAnalysisFoundResponse, BackgroundTaskStatusResponse, NoAttributeFoundResponse, \
    BackgroundTaskFailedResponse, BackgroundTaskRunningResponse, BackgroundTaskFinishedResponse
from api.server_config import API_ATTRIBUTES_COLLECTION_NAME, API_MONGODB_DB_NAME
from api.server_dao.analysis import DAOAsyncAnalysis
from dao.attribute import DAOAsyncAttributePL
from models.attribute import AttributePLInDB

dao_analysis: DAOAsyncAnalysis = DAOAsyncAnalysis()
dao_attribute: DAOAsyncAttributePL = DAOAsyncAttributePL(collection_name=API_ATTRIBUTES_COLLECTION_NAME, db_name=API_MONGODB_DB_NAME)

async def _validate_analysis(
        analysis_id: str) -> Tuple[AnalysisInDB, AttributePLInDB] | NoAnalysisFoundResponse | BackgroundTaskStatusResponse | NoAttributeFoundResponse:
    analysis: Optional[AnalysisInDB] = await dao_analysis.find_one_by_query({'analysis_id': analysis_id})
    if not analysis:
        return NoAnalysisFoundResponse()

    if analysis.status != AnalysisStatus.FINISHED:
        return _handle_analysis_status(analysis)

    attribute: AttributePLInDB = await dao_attribute.find_by_id(analysis.attributes_id)
    if not attribute:
        return NoAttributeFoundResponse()

    return analysis, attribute


def _handle_analysis_status(analysis: AnalysisInDB) -> BackgroundTaskStatusResponse:
    if analysis.status == AnalysisStatus.FAILED:
        return BackgroundTaskFailedResponse(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_hash,
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
            document_id=analysis.document_hash,
            estimated_wait_time=remaining_time
        )
    elif analysis.status == AnalysisStatus.FINISHED:
        return BackgroundTaskFinishedResponse(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_hash,
            estimated_wait_time=0
        )
    else:
        raise Exception(f"Unknown analysis status: {analysis.status}")
