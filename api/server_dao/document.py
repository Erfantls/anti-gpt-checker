from typing import List, Optional

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

    async def find_document_hash_by_query_paginated(
        self,
        query: dict,
        start_index: int = 0,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Find documents matching the query, return only a single field,
        and apply pagination using start_index and limit.

        Args:
            query (dict): MongoDB query filter
            field_name (str): Field to project
            start_index (int): Starting index for pagination
            limit (int): Number of documents to return

        Returns:
            List[Any]: List of values for the specified field
        """
        field_name = "document_hash"
        cursor = self.collection.find(
            filter=query,
            projection={field_name: 1, "_id": 0}
        ).skip(start_index)

        if limit is not None:
            cursor = cursor.limit(limit)
        docs = await cursor.to_list(length=limit)

        return [doc.get(field_name) for doc in docs if field_name in doc]


class DAODocument(DAOBase):
    def __init__(self, collection_name=API_DOCUMENTS_COLLECTION_NAME):
        super().__init__(API_MONGO_CLIENT,
                         API_MONGODB_DB_NAME,
                         collection_name,
                         Document,
                         DocumentInDB)