from dao.base import DAOBase
from config import MONGO_CLIENT, MONGODB_DB_NAME, ATTRIBUTES_COLLECTION_NAME, ASYNC_MONGO_CLIENT
from dao.base_async import DAOBaseAsync
from models.attribute import AttributePL, AttributeEN, AttributePLInDB, AttributeENInDB


class DAOAttributePL(DAOBase):
    def __init__(self, collection_name=ATTRIBUTES_COLLECTION_NAME, db_name=MONGODB_DB_NAME):
        super().__init__(MONGO_CLIENT,
                         db_name,
                         collection_name,
                         AttributePL,
                         AttributePLInDB)

class DAOAsyncAttributePL(DAOBaseAsync):
    def __init__(self, collection_name=ATTRIBUTES_COLLECTION_NAME, db_name=MONGODB_DB_NAME):
        super().__init__(ASYNC_MONGO_CLIENT,
                         db_name,
                         collection_name,
                         AttributePL,
                         AttributePLInDB)

class DAOAttributeEN(DAOBase):
    def __init__(self, collection_name=ATTRIBUTES_COLLECTION_NAME):
        super().__init__(MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         collection_name,
                         AttributeEN,
                         AttributeENInDB)

class DAOAsyncAttributeEN(DAOBaseAsync):
    def __init__(self, collection_name=ATTRIBUTES_COLLECTION_NAME):
        super().__init__(ASYNC_MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         collection_name,
                         AttributeEN,
                         AttributeENInDB)