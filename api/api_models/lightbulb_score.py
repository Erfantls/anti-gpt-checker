from enum import Enum

from pydantic import BaseModel

class LightbulbScoreType(str, Enum):
    BIDIRECTIONAL = "bidirectional" # score [-1,1]
    HUMAN_WRITTEN = "human_written" # score [-1,0]
    LLM_GENERATED = "llm_generated" # score [0,1]

class LightbulbScoreConfig(BaseModel):
    attribute_name: str
    type: LightbulbScoreType

class LightbulbScoreData(LightbulbScoreConfig):
    score: float

