from dao.base import DAOBase
from api.config import API_MONGO_CLIENT, API_MONGODB_DB_NAME, API_ANALYSIS_COLLECTION_NAME
from api.models.analysis import Analysis, An


class DAOAttributePL(DAOBase):
    def __init__(self, collection_name=ATTRIBUTES_COLLECTION_NAME):
        super().__init__(MONGO_CLIENT,
                         MONGODB_DB_NAME,
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