import uvicorn
import hashlib
from datetime import datetime
from bson import ObjectId
from fastapi import FastAPI,  BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from api.models.analysis import Analysis, AnalysisType
from api.models.request import PreprocessedDocumentRequestData
from config import init_all_polish_models

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins if needed
    allow_methods=["*"],  # Adjust this to restrict HTTP methods if needed
    allow_headers=["*"],  # Adjust this to restrict headers if needed
)


@app.post("/add-document")
def post_preprocess_document(preprocessed_document: PreprocessedDocumentRequestData):
    # add document to db
    pass


@app.post("/trigger-analysis")
def trigger_document_analysis(document_id: str, background_tasks: BackgroundTasks, perform_full_analysis: bool = False):
    # generate analysis_id
    analysis_id = hashlib.sha256(f"{document_id}_{datetime.now().isoformat()}".encode()).hexdigest()
    analysis = Analysis(
        analysis_id=analysis_id,
        type=AnalysisType.FULL if perform_full_analysis else AnalysisType.PARTIAL,
        document_id=document_id
    )
    # background_tasks.add_task(analysis, analysis_id)
    return {"message": f"Analysis of document {document_id} has been triggered",
            "analysis_id": str(analysis_id)}

def _perform_analysis(analysis_id: str):
    pass

def _init_server():
    init_all_polish_models()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)