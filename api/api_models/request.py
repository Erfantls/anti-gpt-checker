from typing import List, Optional

from pydantic import BaseModel



class PreprocessedDocumentRequestData(BaseModel):
    document_name: Optional[str]
    document_hash: str
    preprocessed_content: Optional[str]
    filepath: Optional[str]

class LightbulbScoreRequestData(BaseModel):
    attribute_names: List[str]

