from dao.base import DAOBase
from config import MONGO_CLIENT, MONGODB_DB_NAME
from models.email import Email, EmailInDB, \
    EmailGithub, EmailGithubInDB, \
    EmailGmail, EmailGmailInDB, \
    EmailSpamAssassin, EmailSpamAssassinInDB, EmailGenerated, EmailGeneratedInDB


class DAOEmail(DAOBase):
    def __init__(self, collection_name: str):
        super().__init__(MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         collection_name,
                         Email,
                         EmailInDB)


class DAOEmailSpam(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         'email_spam_dataset',
                         Email,
                         EmailInDB)

class DAOEmailGitClass(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         'email_classification_github',
                         EmailGithub,
                         EmailGithubInDB)


class DAOEmailSpamAssassin(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         'email_spam_assassin_dataset',
                         EmailSpamAssassin,
                         EmailSpamAssassinInDB)

class DAOEmailGmail(DAOBase):
    def __init__(self, collection_name: str):
        super().__init__(MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         collection_name,
                         EmailGmail,
                         EmailGmailInDB)


AVAILABLE_EMAIL_DAOS = {
    "spam": DAOEmailSpam(),
    "class_git": DAOEmailGitClass(),
    "spam_assassin": DAOEmailSpamAssassin(),
    "gmail1": DAOEmailGmail('gmail1'),
    "gmail2": DAOEmailGmail('gmail2'),
    "gmail3": DAOEmailGmail('gmail3')
}


class DAORealEmail:
    def __init__(self):
        self.available_daos = AVAILABLE_EMAIL_DAOS

    def find_one(self, db_name: str, query: dict):
        return self.available_daos[db_name].find_one_by_query(query)

    def update_one(self, db_name: str, query: dict, values: dict):
        return self.available_daos[db_name].update_one(query, values)

    def find_one_unknown(self, query: dict):
        for dao in self.available_daos.values():
            result = dao.find_one_by_query(query)
            if result:
                return result
        return None

class DAOEmailGenerated(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_DB_NAME,
                         'email_generated_dataset',
                         EmailGenerated,
                         EmailGeneratedInDB)
