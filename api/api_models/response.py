from enum import Enum
from typing import Union, List

from pydantic import BaseModel

from api.api_models.analysis import AnalysisData, AnalysisStatus


class NoAnalysisFoundResponse(BaseModel):
    type = "no_analysis_found_response"
    message: str = "No analysis found with the specified ID"

class NoAttributeFoundResponse(BaseModel):
    type = "no_attribute_found_response"
    message: str = "No attribute found with the specified ID"

class NoDocumentFoundResponse(BaseModel):
    type = "no_document_found_response"
    message: str = "No document found with the specified ID"

class DocumentWithSpecifiedIDAlreadyExists(BaseModel):
    type = "document_with_specified_id_already_exists"
    message: str = "Document with the specified ID already exists, please use a different ID"


class BackgroundTaskFailedResponse(BaseModel):
    type = "failed_to_extract_data_response"
    message: str = "Failed to extract attributes from the given document"
    status = AnalysisStatus.FAILED
    document_id: str
    estimated_wait_time: int = 0
    analysis_id: str


class BackgroundTaskRunningResponse(BaseModel):
    type = "background_task_running_response"
    message: str = "Background task is running, please wait the given wait-time and call document-analysis-result endpoint with the given analysis_id"
    status = AnalysisStatus.RUNNING
    document_id: str
    estimated_wait_time: int
    analysis_id: str


class BackgroundTaskFinishedResponse(BaseModel):
    type = "background_task_finished_response"
    message: str = "Background task is finished"
    status = AnalysisStatus.FINISHED
    document_id: str
    estimated_wait_time: int = 0
    analysis_id: str

BackgroundTaskStatusResponse = Union[BackgroundTaskRunningResponse, BackgroundTaskFinishedResponse, BackgroundTaskFailedResponse]

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
