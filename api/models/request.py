from typing import List

from pydantic import BaseModel



class PreprocessedDocumentRequestData(BaseModel):
    document_id: str
    preprocessed_text: str

class LightbulbScoreRequestData(BaseModel):
    attribute_names: List[str]

