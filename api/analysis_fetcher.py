import uvicorn

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse

from fastapi.middleware.cors import CORSMiddleware

from api.models.request import LightbulbScoreRequestData
from api.models.response import BackgroundTaskStatusResponse, BackgroundTaskRunningResponse, AnalysisResultsResponse, \
    LightbulbScoreResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins if needed
    allow_methods=["*"],  # Adjust this to restrict HTTP methods if needed
    allow_headers=["*"],  # Adjust this to restrict headers if needed
)


@app.get("/document-analysis-status",
         response_model=BackgroundTaskStatusResponse)
def document_analysis_results(analysis_id: str, background_tasks: BackgroundTasks):
    # running/finished/failed
    pass


@app.get("/document-analysis-result",
         response_model=AnalysisResultsResponse)
def document_analysis_results(analysis_id: str):
    # proper results
    pass

@app.get("/lightbulbs_scores",
         response_model=LightbulbScoreResponse)
def lightbulb_score(analysis_id: str, request_data: LightbulbScoreRequestData):
    for attribute_name in request_data.attribute_names:
        # validate attribute_name
        # get lightbulb score
        pass
    # return list of lightbulb scores

@app.get("/graph-image")
def get_graph_image(analysis_id: str, attribute_name: str):
    # generate image and save it to a file
    image_path = "path/to/image.png"
    return FileResponse(image_path, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)