from enum import Enum
from typing import Optional, Dict

from pydantic import BaseModel

from models.base_mongo_model import MongoDBModel, MongoObjectId


class LightbulbScoreType(str, Enum):
    BIDIRECTIONAL = "bidirectional" # score [-1,1]
    HUMAN_WRITTEN = "human_written" # score [-1,0]
    LLM_GENERATED = "llm_generated" # score [0,1]

class LightbulbScoreConfig(BaseModel):
    attribute_name: str
    type: LightbulbScoreType
    type_direction: Optional[int] = None
    max_value: Optional[float]
    min_value: Optional[float]
    feature_rank: Optional[int] = None

class LightbulbScoreData(LightbulbScoreConfig):
    score: float
    raw_score: float

class LightbulbScores(BaseModel):
    attribute_id: MongoObjectId
    is_chunk_attribute: bool = False
    identifier: Optional[int] = None
    lightbulb_scores_dict: Dict[str, LightbulbScoreData]

class LightbulbScoresInDB(MongoDBModel, LightbulbScores):
    pass

