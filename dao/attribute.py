from dao.base import DAOBase
from config import MONGO_CLIENT, MONGODB_DB_NAME
from models.attribute import Attribute, AttributeInDB


class DAOAttribute(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         'attributes',
                         Attribute,
                         AttributeInDB)