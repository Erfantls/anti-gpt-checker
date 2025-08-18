from typing import List, Optional

from pydantic import BaseModel

from api.api_models.document import DocumentStatus


class PreprocessedDocumentRequestData(BaseModel):
    document_name: Optional[str]
    document_hash: str
    plaintext_content: Optional[str]
    filepath: Optional[str]
    document_status: Optional[DocumentStatus]



class LightbulbScoreRequestData(BaseModel):
    attribute_names: List[str]

