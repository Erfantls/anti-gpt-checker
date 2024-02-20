from enum import Enum
from typing import Union, List, Literal, Annotated, Dict

from pydantic import BaseModel, Field

from models.base_mongo_model import MongoObjectId, MongoDBModel
from models.stylometrix_metrics import StyloMetrixMetrics


class MetricNameEnum(str, Enum):
    PERPLEXITY = "perplexity"
    BURSTINESS = "burstiness"
    AVERAGE_WORD_LENGTH = "average_word_length"
    AVERAGE_SENTENCE_LENGTH = "average_sentence_length"
    STYLOMETRIX_METRICS = "stylometrix_metrics"
    POS_ENG_TAGS = "pos_eng_tags"
    SENTIMENT_ENG = "sentiment_eng"
    PUNCTUATION = "punctuation"


class PerplexityMetric(BaseModel):
    name: Literal[MetricNameEnum.PERPLEXITY] = MetricNameEnum.PERPLEXITY.value
    value: float

class BurstinessMetric(BaseModel):
    name: Literal[MetricNameEnum.BURSTINESS] = MetricNameEnum.BURSTINESS.value
    value: float

class AverageSentenceLengthMetric(BaseModel):
    name: Literal[MetricNameEnum.AVERAGE_SENTENCE_LENGTH] = MetricNameEnum.AVERAGE_SENTENCE_LENGTH.value
    value: float

class AverageWordLengthMetric(BaseModel):
    name: Literal[MetricNameEnum.AVERAGE_WORD_LENGTH] = MetricNameEnum.AVERAGE_WORD_LENGTH.value
    value: float

class StyloMetrixMetric(BaseModel):
    name: Literal[MetricNameEnum.STYLOMETRIX_METRICS] = MetricNameEnum.STYLOMETRIX_METRICS.value
    value: StyloMetrixMetrics

class PosEngTagMetric(BaseModel):
    name: Literal[MetricNameEnum.POS_ENG_TAGS] = MetricNameEnum.POS_ENG_TAGS.value
    value: Dict[str, int]

class SentimentEngMetric(BaseModel):
    name: Literal[MetricNameEnum.SENTIMENT_ENG] = MetricNameEnum.SENTIMENT_ENG.value
    value: Dict[str, float]

class PunctuationMetric(BaseModel):
    name: Literal[MetricNameEnum.PUNCTUATION] = MetricNameEnum.PUNCTUATION.value
    value: Dict[str, int]

class BaseMetric(BaseModel):
    __root__: Annotated[Union[
                PerplexityMetric,
                BurstinessMetric,
                AverageSentenceLengthMetric,
                AverageWordLengthMetric,
                StyloMetrixMetric
                ],
                Field(..., discriminator='name')]

class AttributeBase(BaseModel):
    referenced_db_name: str
    referenced_doc_id: MongoObjectId
    metrics: List[BaseMetric]

class Attribute(AttributeBase):
    pass

class AttributeInDB(MongoDBModel, AttributeBase):
    pass
