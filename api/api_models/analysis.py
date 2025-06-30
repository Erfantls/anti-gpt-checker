from datetime import datetime
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel

from models.attribute import AttributeInDB
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
    attributes_id: Optional[MongoObjectId] = None
    estimated_wait_time: int
    start_time: datetime
    error_message: Optional[str] = None


class AnalysisInDB(MongoDBModel, Analysis):
    pass


class AnalysisData(BaseModel):
    analysis_id: str
    document_id: str
    full_features: dict

    @staticmethod
    def from_analysis_and_attribute(analysis_in_db: AnalysisInDB, attribute_in_db: AttributeInDB):
        # I dont understand why this is needed, but if its not done, the id is ObjectId and not MongoObjectId and FastAPI cannot serialize it
        attribute_dict = attribute_in_db.dict()
        attribute_dict["id"] = str(attribute_in_db.id)
        attribute_dict["referenced_doc_id"] = str(attribute_in_db.referenced_doc_id)
        return AnalysisData(
            analysis_id=analysis_in_db.analysis_id,
            document_id=analysis_in_db.document_id,
            full_features=attribute_dict
        )
