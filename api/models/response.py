from enum import Enum
from typing import Union, List

from pydantic import BaseModel

from api.models.analysis import AnalysisData
from models.base_mongo_model import MongoObjectId


class NoReviewsFoundResponse(BaseModel):
    type = "no_reviews_found_response"
    message: str = "No reviews found on the specified place"

class FailedToExtractData(BaseModel):
    type = "failed_to_extract_data_response"
    message: str = "Failed to extract attributes from the given document, if you are sure that the document is not malformed, please call renew-markers endpoint first"

class BackgroundTaskStillRunningResponse(BaseModel):
    type = "background_task_still_running_response"
    message: str = "Background task is still running, please wait the given wait-time and then call this endpoint again"
    document_id: str
    estimated_wait_time: int
    analysis_id = None

class BackgroundTaskRunningResponse(BaseModel):
    type = "background_task_running_response"
    message: str = "Background task is running, please wait the given wait-time and call check-results endpoint with the given document_id"
    document_id: str
    estimated_wait_time: int
    analysis_id = None

class BackgroundTaskFinishedResponse(BaseModel):
    type = "background_task_finished_response"
    message: str = "Background task is finished"
    document_id: str
    estimated_wait_time: int = 0
    analysis_id: MongoObjectId

BackgroundTaskStatusResponse = Union[BackgroundTaskRunningResponse, BackgroundTaskFinishedResponse, BackgroundTaskStillRunningResponse]

class AnalysisResultsResponse(BaseModel):
    type = "analysis_results_response"
    message: str = "Analysis results"
    analysis_data: AnalysisData

class LightbulbScoreType(str, Enum):
    BIDIRECTIONAL = "bidirectional" # score [-1,1]
    HUMAN_WRITTEN = "human_written" # score [-1,0]
    LLM_GENERATED = "llm_generated" # score [0,1]

class LightbulbScoreData(BaseModel):
    attribute_name: str
    type: LightbulbScoreType
    score: float

class LightbulbScoreResponse(BaseModel):
    type = "lightbulb_score_response"
    lightbulb_scores: List[LightbulbScoreData]
