from api.server_config import API_ASYNC_MONGO_CLIENT, API_MONGODB_DB_NAME, API_USERS_COLLECTION_NAME, \
    API_MONGO_CLIENT
from api.api_models.user import User, UserInDB
from dao.base import DAOBase
from dao.base_async import DAOBaseAsync


class DAOAsyncUser(DAOBaseAsync):
    def __init__(self, collection_name=API_USERS_COLLECTION_NAME):
        super().__init__(API_ASYNC_MONGO_CLIENT,
                         API_MONGODB_DB_NAME,
                         collection_name,
                         User,
                         UserInDB)

class DAOUser(DAOBase):
    def __init__(self, collection_name=API_USERS_COLLECTION_NAME):
        super().__init__(API_MONGO_CLIENT,
                         API_MONGODB_DB_NAME,
                         collection_name,
                         User,
                         UserInDB)