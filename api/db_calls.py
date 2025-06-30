from typing import Optional

from fastapi import Depends, APIRouter, HTTPException, Body
from bson import ObjectId


from api.server_config import API_ATTRIBUTES_COLLECTION_NAME, API_DEBUG, API_MONGODB_DB_NAME, API_HISTOGRAMS_PATH

from api.api_models.user import User, UserInDB, UserRole
from api.api_models.document import Document, DocumentInDB
from api.api_models.analysis import Analysis, AnalysisInDB, AnalysisStatus, AnalysisData
from api.api_models.response import BackgroundTaskStatusResponse, BackgroundTaskRunningResponse, NoUserFoundResponse, \
    DocumentsOfUserResponse, NoDocumentFoundResponse, AnalysesOfDocumentsResponse

from api.security import verify_token, generate_salt, hash_password_with_salt

from api.server_dao.analysis import DAOAsyncAnalysis
from api.server_dao.document import DAOAsyncDocument
from api.server_dao.user import DAOAsyncUser
from dao.attribute import DAOAsyncAttributePL
from models.attribute import AttributePLInDB

router = APIRouter()
dao_user: DAOAsyncUser = DAOAsyncUser()
dao_document: DAOAsyncDocument = DAOAsyncDocument()
dao_analysis: DAOAsyncAnalysis = DAOAsyncAnalysis()
dao_attribute: DAOAsyncAttributePL = DAOAsyncAttributePL(collection_name=API_ATTRIBUTES_COLLECTION_NAME, db_name=API_MONGODB_DB_NAME)

@router.get("/get-user-by-email",
            response_model=UserInDB | NoUserFoundResponse)
async def document_analysis_results(user_email: str, _: bool = Depends(verify_token) if not API_DEBUG else True):
    user: Optional[UserInDB] = await dao_user.find_one_by_query({'email': user_email})
    if not user:
        return NoUserFoundResponse()

    return user


@router.get("/get-user-by-id",
            response_model=UserInDB | NoUserFoundResponse)
async def get_user_by_id(user_id: str, _: bool = Depends(verify_token) if not API_DEBUG else True):
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    user: Optional[UserInDB] = await dao_user.find_by_id(user_id)
    if not user:
        return NoUserFoundResponse()

    return user

@router.get("/get-user-by-username",
            response_model=UserInDB | NoUserFoundResponse)
async def get_user_by_username(username: str, _: bool = Depends(verify_token) if not API_DEBUG else True):
    user: Optional[UserInDB] = await dao_user.find_one_by_query({'username': username})
    if not user:
        return NoUserFoundResponse()

    return user


@router.post("/create-user", response_model=UserInDB)
async def create_user(
    username: str = Body(...),
    email: str = Body(...),
    password: str = Body(...),
    _: bool = Depends(verify_token) if not API_DEBUG else True
):
    existing_user = await dao_user.find_one_by_query({'email': email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User with this email already exists")
    existing_user = await dao_user.find_one_by_query({'username': username})
    if existing_user:
        raise HTTPException(status_code=400, detail="User with this username already exists")

    salt = generate_salt()
    password_hash = hash_password_with_salt(password, salt)

    user = User(
        username=username,
        email=email,
        password_hash=password_hash,
        password_salt=salt,
        role=UserRole.USER
    )

    insert_result = await dao_user.insert_one(user)
    user_in_db = await dao_user.find_by_id(insert_result)

    return user_in_db

@router.get("/get-documents-of-user",
            response_model=DocumentsOfUserResponse | NoUserFoundResponse)
async def get_documents_of_username(username: str, _: bool = Depends(verify_token) if not API_DEBUG else True):
    user: Optional[UserInDB] = await dao_user.find_one_by_query({'username': username})
    if not user:
        return NoUserFoundResponse()

    documents: list[DocumentInDB] = await dao_document.find_many_by_query({'user_id': user.id})

    return DocumentsOfUserResponse(documents=documents)


@router.get("/get-document",
            response_model=DocumentInDB | NoDocumentFoundResponse)
async def get_document(document_id: str, _: bool = Depends(verify_token) if not API_DEBUG else True):
    document: Optional[DocumentInDB] = await dao_document.find_one_by_query({'document_id': document_id})
    if not document_id:
        return NoDocumentFoundResponse()

    return document

@router.get("/get-analysis-of-document",
            response_model=AnalysesOfDocumentsResponse)
async def get_analyses_of_document(document_id: str, _: bool = Depends(verify_token) if not API_DEBUG else True):
    analyses: list[AnalysisInDB] = await dao_analysis.find_many_by_query({'document_id': document_id})

    analyses_data = []
    for analysis in analyses:
        if analysis.status != AnalysisStatus.FINISHED:
            analyses_data.append(AnalysisData(analysis_id=analysis.analysis_id, document_id=analysis.document_id, full_features={}))
            continue

        attribute: AttributePLInDB = await dao_attribute.find_by_id(analysis.attributes_id)
        if not attribute:
            analyses_data.append(
                AnalysisData(analysis_id=analysis.analysis_id, document_id=analysis.document_id, full_features={}))
            continue

        analyses_data.append(AnalysisData.from_analysis_and_attribute(analysis, attribute))

    return AnalysesOfDocumentsResponse(analyses=analyses_data)


