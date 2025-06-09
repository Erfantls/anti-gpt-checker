import uvicorn
import hashlib
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware

from analysis.attribute_retriving import perform_full_analysis
from analysis.nlp_transformations import preprocess_text
from api.config import API_ATTRIBUTES_COLLECTION_NAME, API_DOCUMENTS_COLLECTION_NAME
from api.dao.analysis import DAOAsyncAnalysis
from api.dao.document import DAOAsyncDocument
from api.models.analysis import Analysis, AnalysisType, AnalysisStatus
from api.models.document import DocumentInDB, Document
from api.models.request import PreprocessedDocumentRequestData
from api.security import verify_token
from config import init_all_polish_models
from dao.attribute import DAOAsyncAttributePL
from models.attribute import AttributePL

app = FastAPI()

dao_analysis: DAOAsyncAnalysis = DAOAsyncAnalysis()
dao_document: DAOAsyncDocument = DAOAsyncDocument()
dao_attribute: DAOAsyncAttributePL = DAOAsyncAttributePL(collection_name=API_ATTRIBUTES_COLLECTION_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins if needed
    allow_methods=["*"],  # Adjust this to restrict HTTP methods if needed
    allow_headers=["*"],  # Adjust this to restrict headers if needed
)


@app.post("/add-document")
async def post_preprocess_document(preprocessed_document: PreprocessedDocumentRequestData,
                                   _: bool = Depends(verify_token)):
    document = Document(
        document_id=preprocessed_document.document_id,
        plaintext_content=preprocessed_document.plaintext_content,
        filepath=preprocessed_document.filepath
    )
    await dao_document.insert_one(document)
    return {"message": f"Document has been inserted"}


@app.post("/trigger-analysis")
async def trigger_document_analysis(document_id: str, background_tasks: BackgroundTasks,
                                    perform_full_analysis: bool = False, _: bool = Depends(verify_token)):
    # generate analysis_id
    analysis_id = hashlib.sha256(f"{document_id}_{datetime.now().isoformat()}".encode()).hexdigest()
    analysis = Analysis(
        analysis_id=analysis_id,
        type=AnalysisType.FULL if perform_full_analysis else AnalysisType.PARTIAL,
        status=AnalysisStatus.RUNNING,
        document_id=document_id,
        estimated_wait_time=30,
        start_time=datetime.now()
    )
    await dao_analysis.insert_one(analysis)
    background_tasks.add_task(_perform_analysis, analysis_id, document_id)
    return {"message": f"Analysis of document {document_id} has been triggered",
            "analysis_id": str(analysis_id)}


async def _perform_analysis(analysis_id: str, document_id):
    document: DocumentInDB = await dao_document.find_one_by_query({'document_id': document_id})
    try:
        text_to_analyse = preprocess_text(document.plaintext_content)
        analysis_result = perform_full_analysis(text_to_analyse, 'pl')
        attribute_to_insert = AttributePL(
            referenced_db_name=API_DOCUMENTS_COLLECTION_NAME,
            referenced_doc_id=document.id,
            language="pl",
            is_generated=None,
            is_personal=None,
            **analysis_result.dict()
        )
        await dao_attribute.insert_one(attribute_to_insert)
        await dao_analysis.update_one({'analysis_id': analysis_id}, {'$set': {'status': AnalysisStatus.FINISHED}})
    except:
        await dao_analysis.update_one({'analysis_id': analysis_id}, {'$set': {'status': AnalysisStatus.FAILED}})


def _init_server():
    init_all_polish_models()


if __name__ == "__main__":
    _init_server()
    uvicorn.run(app, host="0.0.0.0", port=8000)
