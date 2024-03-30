from typing import Dict, Optional

from pydantic import BaseModel

from models.base_mongo_model import MongoObjectId, MongoDBModel
from models.stylometrix_metrics import StyloMetrixMetrics
class AttributeBase(BaseModel):
    referenced_db_name: str #V
    referenced_doc_id: MongoObjectId #V
    perplexity: Optional[float] #V
    perplexity_base: Optional[float] #V
    burstiness: Optional[float] #V

    average_sentence_word_length: Optional[float] #P
    standard_deviation_sentence_word_length: Optional[float] #P
    variance_sentence_word_length: Optional[float] #P

    standard_deviation_sentence_char_length: Optional[float] #P
    variance_sentence_char_length: Optional[float] #P
    average_sentence_char_length: Optional[float] #P

    average_word_char_length: Optional[float] #P
    punctuation: Optional[int] #P
    number_of_sentences: Optional[int] #P
    number_of_words: Optional[int] #V
    number_of_characters: Optional[int] #V
    stylometrix_metrics: Optional[StyloMetrixMetrics]
    pos_eng_tags: Optional[Dict[str, int]]
    sentiment_eng: Optional[Dict[str, float]]


class Attribute(AttributeBase):
    pass

class AttributeInDB(MongoDBModel, AttributeBase):
    pass
