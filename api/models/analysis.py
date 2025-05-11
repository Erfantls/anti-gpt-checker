from enum import Enum
from typing import Optional, List

from pydantic import BaseModel

from models.attribute import AttributeInDB, PartialAttribute
from models.base_mongo_model import MongoDBModel


class AnalysisType(str, Enum):
    FULL = "full"
    PARTIAL = "partial"

class Analysis(BaseModel):
    analysis_id: str
    type: AnalysisType
    document_id: str


class DocumentInDB(MongoDBModel, Analysis):
    pass


class AnalysisData(BaseModel):
    analysis_id: str
    document_id: str
    full_features: AttributeInDB
    partial_features: Optional[List[PartialAttribute]] = None
