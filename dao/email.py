from dao.base import DAOBase
from config import MONGO_CLIENT, MONGODB_DB_NAME, MONGODB_EMAILS_SRC_COLLECTIONS
from models.email import Email, EmailInDB, EmailGithubDataset, EmailGithubDatasetInDB


class DAOEmail(DAOBase):
    def __init__(self, collection_name: str):
        super().__init__(MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         collection_name,
                         Email,
                         EmailInDB)


class DAOEmailGitClass(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         MONGODB_EMAILS_SRC_COLLECTIONS["email_class_git"],
                         EmailGithubDataset,
                         EmailGithubDatasetInDB)
