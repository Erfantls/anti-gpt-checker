from enum import Enum
from typing import Union, List

from pydantic import BaseModel

from api.api_models.analysis import AnalysisData, AnalysisStatus
from api.api_models.document import DocumentInDB


class NoAnalysisFoundResponse(BaseModel):
    type: str = "no_analysis_found_response"
    message: str = "No analysis found with the specified ID"

class NoAttributeFoundResponse(BaseModel):
    type: str = "no_attribute_found_response"
    message: str = "No attribute found with the specified ID"

class NoDocumentFoundResponse(BaseModel):
    type: str = "no_document_found_response"
    message: str = "No document found with the specified ID"

class NoUserFoundResponse(BaseModel):
    type: str = "no_user_found_response"
    message: str = "No user found with the specified ID"

class DocumentWithSpecifiedIDAlreadyExists(BaseModel):
    type: str = "document_with_specified_id_already_exists"
    message: str = "Document with the specified ID already exists, please use a different ID"


class BackgroundTaskFailedResponse(BaseModel):
    type: str = "failed_to_extract_data_response"
    message: str = "Failed to extract attributes from the given document"
    status: AnalysisStatus = AnalysisStatus.FAILED
    document_id: str
    estimated_wait_time: int = 0
    analysis_id: str


class BackgroundTaskRunningResponse(BaseModel):
    type: str = "background_task_running_response"
    message: str = "Background task is running, please wait the given wait-time and call document-analysis-result endpoint with the given analysis_id"
    status: AnalysisStatus = AnalysisStatus.RUNNING
    document_id: str
    estimated_wait_time: int
    analysis_id: str


class BackgroundTaskFinishedResponse(BaseModel):
    type: str = "background_task_finished_response"
    message: str = "Background task is finished"
    status: AnalysisStatus = AnalysisStatus.FINISHED
    document_id: str
    estimated_wait_time: int = 0
    analysis_id: str

BackgroundTaskStatusResponse = Union[BackgroundTaskRunningResponse, BackgroundTaskFinishedResponse, BackgroundTaskFailedResponse]

class AnalysisResultsResponse(BaseModel):
    type: str = "analysis_results_response"
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
    type: str = "lightbulb_score_response"
    lightbulb_scores: List[LightbulbScoreData]

class DocumentsOfUserResponse(BaseModel):
    type: str = "documents_of_user_response"
    message: str = "Documents of the user"
    documents: List[DocumentInDB]  # List of documents

class AnalysesOfDocumentsResponse(BaseModel):
    type: str = "analyses_of_documents_response"
    message: str = "Analyses of the documents"
    analyses: List[AnalysisData]  # List of AnalysisData objects