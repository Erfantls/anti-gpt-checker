from api.server_config import API_ASYNC_MONGO_CLIENT, API_MONGODB_DB_NAME, API_MONGO_CLIENT, API_LIGHTBULBS_SCORES_COLLECTION_NAME
from api.api_models.lightbulb_score import LightbulbScores, LightbulbScoresInDB
from dao.base import DAOBase
from dao.base_async import DAOBaseAsync


class DAOAsyncLightbulbScore(DAOBaseAsync):
    def __init__(self, collection_name=API_LIGHTBULBS_SCORES_COLLECTION_NAME):
        super().__init__(API_ASYNC_MONGO_CLIENT,
                         API_MONGODB_DB_NAME,
                         collection_name,
                         LightbulbScores,
                         LightbulbScoresInDB)

class DAOLightbulbScore(DAOBase):
    def __init__(self, collection_name=API_LIGHTBULBS_SCORES_COLLECTION_NAME):
        super().__init__(API_MONGO_CLIENT,
                         API_MONGODB_DB_NAME,
                         collection_name,
                         LightbulbScores,
                         LightbulbScoresInDB)