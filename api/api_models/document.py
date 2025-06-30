from enum import Enum
from typing import Optional

from pydantic import BaseModel

from models.base_mongo_model import MongoDBModel, MongoObjectId


class DocumentStatus(str, Enum):
    PREPROCESS_RUNNING = "preprocess_running"
    READY_FOR_ANALYSIS = "ready_for_analysis"


class Document(BaseModel):
    plaintext_content: Optional[str]
    filepath: str
    document_id: Optional[str]
    document_status: Optional[DocumentStatus] = DocumentStatus.READY_FOR_ANALYSIS
    owner_id: Optional[MongoObjectId] = None


class DocumentInDB(MongoDBModel, Document):
    pass