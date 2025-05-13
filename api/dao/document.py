from dao.base import DAOBase
from api.config import API_MONGO_CLIENT, API_MONGODB_DB_NAME, API_DOCUMENTS_COLLECTION_NAME
from api.models.document import Document, DocumentInDB


class DAODocument(DAOBase):
    def __init__(self, collection_name=API_DOCUMENTS_COLLECTION_NAME):
        super().__init__(API_MONGO_CLIENT,
                         API_MONGODB_DB_NAME,
                         collection_name,
                         Document,
                         DocumentInDB)
