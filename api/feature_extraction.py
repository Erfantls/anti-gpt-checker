import uvicorn
from bson import ObjectId
from fastapi import FastAPI,  BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from api.models.request import PreprocessedDocumentRequestData

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
def trigger_document_analysis(document_id: str, background_tasks: BackgroundTasks):
    # generate analysis_id
    analysis_id = ObjectId()
    # background_tasks.add_task(analysis, analysis_id)
    return {"message": f"Analysis of document {document_id} has been triggered",
            "analysis_id": str(analysis_id)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)