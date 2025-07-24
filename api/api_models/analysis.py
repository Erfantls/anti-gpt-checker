from datetime import datetime
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel

from models.attribute import AttributeInDB
from models.base_mongo_model import MongoDBModel, MongoObjectId


class AnalysisType(str, Enum):
    DOCUMENT_LEVEL = "document_level"
    CHUNK_LEVEL = "chunk_level"

class AnalysisStatus(str, Enum):
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"

class Analysis(BaseModel):
    analysis_id: str
    type: AnalysisType
    status: AnalysisStatus
    document_hash: str
    attributes_id: Optional[MongoObjectId] = None
    estimated_wait_time: int
    start_time: datetime
    error_message: Optional[str] = None


class AnalysisInDB(MongoDBModel, Analysis):
    pass


class AnalysisData(BaseModel):
    analysis_id: str
    document_hash: str
    full_features: Optional[AttributeInDB]

    @staticmethod
    def from_analysis_and_attribute(analysis_in_db: AnalysisInDB, attribute_in_db: Optional[AttributeInDB]):
        return AnalysisData(
            analysis_id=analysis_in_db.analysis_id,
            document_hash=analysis_in_db.document_hash,
            full_features=attribute_in_db
        )
