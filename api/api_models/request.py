from typing import List, Optional

from pydantic import BaseModel



class PreprocessedDocumentRequestData(BaseModel):
    document_name: str
    preprocessed_content: str
    filepath: Optional[str]

class LightbulbScoreRequestData(BaseModel):
    attribute_names: List[str]

