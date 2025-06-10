from pydantic import BaseModel

from models.base_mongo_model import MongoDBModel

class Document(BaseModel):
    plaintext_content: str
    filepath: str
    document_id: str


class DocumentInDB(MongoDBModel, Document):
    pass