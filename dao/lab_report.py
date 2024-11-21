from dao.base import DAOBase
from config import MONGO_CLIENT, MONGODB_DB_NAME, LAB_REPORTS_COLLECTION_NAME
from models.lab_report import LabReport, LabReportInDB


class DAOLabReport(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         LAB_REPORTS_COLLECTION_NAME,
                         LabReport,
                         LabReportInDB)