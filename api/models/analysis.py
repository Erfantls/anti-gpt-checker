from enum import Enum
from typing import Optional, List

from pydantic import BaseModel

from models.attribute import AttributeInDB, PartialAttribute
from models.base_mongo_model import MongoDBModel, MongoObjectId


class AnalysisType(str, Enum):
    FULL = "full"
    PARTIAL = "partial"

class AnalysisStatus(str, Enum):
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"

class Analysis(BaseModel):
    analysis_id: str
    type: AnalysisType
    status: AnalysisStatus
    document_id: str
    features_id: MongoObjectId


class AnalysisInDB(MongoDBModel, Analysis):
    pass


class AnalysisData(BaseModel):
    analysis_id: str
    document_id: str
    full_features: AttributeInDB
    partial_features: Optional[List[PartialAttribute]] = None
