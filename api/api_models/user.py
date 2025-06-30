from enum import Enum

from pydantic import BaseModel

from models.base_mongo_model import MongoDBModel

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"

class User(BaseModel):
    username: str
    email: str
    password_hash: str
    password_salt: str
    role: UserRole = UserRole.USER


class UserInDB(MongoDBModel, User):
    pass