from api.config import API_MONGO_CLIENT, API_MONGODB_DB_NAME, API_ANALYSIS_COLLECTION_NAME
from api.models.analysis import Analysis, AnalysisInDB
from dao.base_async import DAOBaseAsync


class DAOAsyncAnalysis(DAOBaseAsync):
    def __init__(self, collection_name=API_ANALYSIS_COLLECTION_NAME):
        super().__init__(API_MONGO_CLIENT,
                         API_MONGODB_DB_NAME,
                         collection_name,
                         Analysis,
                         AnalysisInDB)
