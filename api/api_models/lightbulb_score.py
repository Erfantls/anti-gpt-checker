from enum import Enum
from typing import Optional

from pydantic import BaseModel

class LightbulbScoreType(str, Enum):
    BIDIRECTIONAL = "bidirectional" # score [-1,1]
    HUMAN_WRITTEN = "human_written" # score [-1,0]
    LLM_GENERATED = "llm_generated" # score [0,1]

class LightbulbScoreConfig(BaseModel):
    attribute_name: str
    type: LightbulbScoreType
    max_value: Optional[float]
    min_value: Optional[float]
    feature_rank: Optional[int] = None

class LightbulbScoreData(LightbulbScoreConfig):
    score: float
    raw_score: float

