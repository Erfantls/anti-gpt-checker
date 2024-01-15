from enum import Enum
from typing import Union, List, Optional

from pydantic import BaseModel
from datetime import datetime

from models.base_mongo_model import MongoDBModel


class GithubClassEnums(str, Enum):
    CALENDAR = "calendar"
    PERSONAL = "personal"
    MEETINGS = "meetings"


class EmailBase(BaseModel):
    from_address: Optional[str]
    to_address: Optional[Union[str, List[str]]]
    date: Optional[datetime]
    subject: Optional[str]
    body: Optional[str]
    is_html: bool = False
    is_spam: Optional[bool]
    is_ai_generated: Optional[bool]

class Email(EmailBase):
    pass

class EmailInDB(MongoDBModel, EmailBase):
    pass

class EmailGithubDataset(EmailBase):
    inner_classification: GithubClassEnums
class EmailPayload(BaseModel):
    is_html: Optional[bool]
    body: Optional[Union[str,List['EmailPayload']]]

class EmailGmail(EmailBase):
    is_html: Optional[bool] = False
    from_name: Optional[str]
    email_labels: Optional[str]
    detected_language: Optional[str]
    text_plain: Optional[str]

class EmailGmailInDB(MongoDBModel, EmailGmail):
    pass

class EmailGithubDatasetInDB(MongoDBModel, EmailGithubDataset):
    pass