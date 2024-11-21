from dao.base import DAOBase
from config import MONGO_CLIENT, MONGODB_DB_NAME, ATTRIBUTES_COLLECTION_NAME
from models.attribute import AttributePL, AttributeEN, AttributePLInDB, AttributeENInDB


class DAOAttributePL(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         ATTRIBUTES_COLLECTION_NAME,
                         AttributePL,
                         AttributePLInDB)

class DAOAttributeEN(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         ATTRIBUTES_COLLECTION_NAME,
                         AttributeEN,
                         AttributeENInDB)