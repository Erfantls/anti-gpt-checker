from api.server_config import API_ASYNC_MONGO_CLIENT, API_MONGODB_DB_NAME, API_DOCUMENTS_COLLECTION_NAME, \
    API_MONGO_CLIENT
from api.api_models.document import Document, DocumentInDB
from dao.base import DAOBase
from dao.base_async import DAOBaseAsync


class DAOAsyncDocument(DAOBaseAsync):
    def __init__(self, collection_name=API_DOCUMENTS_COLLECTION_NAME):
        super().__init__(API_ASYNC_MONGO_CLIENT,
                         API_MONGODB_DB_NAME,
                         collection_name,
                         Document,
                         DocumentInDB)

class DAODocument(DAOBase):
    def __init__(self, collection_name=API_DOCUMENTS_COLLECTION_NAME):
        super().__init__(API_MONGO_CLIENT,
                         API_MONGODB_DB_NAME,
                         collection_name,
                         Document,
                         DocumentInDB)