from dao.base import DAOBase
from config import MONGO_CLIENT, MONGODB_DB_NAME, LAB_REPORTS_COLLECTION_NAME, ASYNC_MONGO_CLIENT
from dao.base_async import DAOBaseAsync
from models.lab_report import LabReport, LabReportInDB


class DAOLabReport(DAOBase):
    def __init__(self, collection_name=LAB_REPORTS_COLLECTION_NAME):
        super().__init__(MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         collection_name,
                         LabReport,
                         LabReportInDB)

class DAOAsyncLabReport(DAOBaseAsync):
    def __init__(self, collection_name=LAB_REPORTS_COLLECTION_NAME):
        super().__init__(ASYNC_MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         collection_name,
                         LabReport,
                         LabReportInDB)