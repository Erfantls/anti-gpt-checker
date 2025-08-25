from typing import Type, List, Optional, Any

from bson import ObjectId
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase,
    AsyncIOMotorCollection,
    AsyncIOMotorCursor,
)
from pydantic import BaseModel


class DAOBaseAsync:
    client: AsyncIOMotorClient
    db: AsyncIOMotorDatabase
    collection_name: str
    collection: AsyncIOMotorCollection

    base_model: Type[BaseModel]
    model_in_db: Type[BaseModel]

    # ──────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        client: AsyncIOMotorClient,
        db_name: str,
        collection_name: str,
        base_model: Type[BaseModel],
        model_in_db: Type[BaseModel],
    ):
        self.client = client
        self.db = self.client[db_name]
        self.collection_name = collection_name
        self.collection = self.db[collection_name]
        self.base_model = base_model
        self.model_in_db = model_in_db

    # ──────────────────────────────────────────────────────────────────────────
    # READ
    # ──────────────────────────────────────────────────────────────────────────
    async def find_all(self) -> List[BaseModel]:
        cursor: AsyncIOMotorCursor = self.collection.find({})
        docs = await cursor.to_list(length=None)
        return [self.model_in_db(**doc) for doc in docs]

    async def find_by_id(self, _id: ObjectId | str) -> Optional[BaseModel]:
        if not isinstance(_id, ObjectId):
            _id = ObjectId(_id)
        doc = await self.collection.find_one({"_id": _id})
        return self.model_in_db(**doc) if doc else None

    async def find_one_by_query(self, query: dict, projections: Optional[List[str]] = None, sort: Optional[List] = None) -> Optional[BaseModel]:
        kwargs = {}
        if sort is not None:
            kwargs["sort"] = sort
        if projections is not None:
            projections_dict = {projection: 1 for projection in projections}
            kwargs["projection"] = projections_dict
        doc = await self.collection.find_one(query, **kwargs)
        return self.model_in_db(**doc) if doc else None

    async def find_one_by_query_raw(self, query: dict) -> Optional[dict]:
        return await self.collection.find_one(query)

    async def find_many_by_query(self, query: dict) -> List[BaseModel]:
        cursor: AsyncIOMotorCursor = self.collection.find(query)
        docs = await cursor.to_list(length=None)
        return [self.model_in_db(**doc) for doc in docs]

    # ──────────────────────────────────────────────────────────────────────────
    # CREATE
    # ──────────────────────────────────────────────────────────────────────────
    async def insert_one(self, obj: BaseModel | dict) -> ObjectId:
        if isinstance(obj, BaseModel):
            obj = obj.dict()
        result = await self.collection.insert_one(obj)
        return result.inserted_id

    async def insert_many(self, obj_list: List[BaseModel | dict]) -> List[ObjectId]:
        dicts: List[dict] = [
            (o.dict() if isinstance(o, BaseModel) else o) for o in obj_list
        ]
        result = await self.collection.insert_many(dicts)
        return result.inserted_ids

    # ──────────────────────────────────────────────────────────────────────────
    # UPDATE / REPLACE
    # ──────────────────────────────────────────────────────────────────────────
    async def replace_one(
        self, field_name: str, value: Any, obj: BaseModel | dict
    ) -> bool:
        if isinstance(obj, BaseModel):
            obj = obj.dict()
        result = await self.collection.replace_one({field_name: value}, obj)
        return result.acknowledged

    async def update_one(self, query: dict, values: dict) -> int:
        result = await self.collection.update_one(query, values)
        return result.matched_count

    # ──────────────────────────────────────────────────────────────────────────
    # DELETE
    # ──────────────────────────────────────────────────────────────────────────
    async def delete_one(self, query: dict) -> int:
        result = await self.collection.delete_one(query)
        return result.deleted_count

    async def delete_many(self, query: dict) -> int:
        result = await self.collection.delete_many(query)
        return result.deleted_count
