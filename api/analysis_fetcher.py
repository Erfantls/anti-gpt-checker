from datetime import timedelta, datetime
from typing import Optional

import uvicorn

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse

from fastapi.middleware.cors import CORSMiddleware

from api.config import API_ATTRIBUTES_COLLECTION_NAME, API_ATTRIBUTES_REFERENCE_COLLECTION_NAME
from api.dao.analysis import DAOAnalysis
from api.dao.document import DAODocument
from api.models.analysis import AnalysisInDB, AnalysisStatus, AnalysisData
from api.models.request import LightbulbScoreRequestData
from api.models.response import BackgroundTaskStatusResponse, BackgroundTaskRunningResponse, AnalysisResultsResponse, \
    LightbulbScoreResponse, NoAnalysisFoundResponse, BackgroundTaskFailedResponse, NoAttributeFoundResponse, \
    LightbulbScoreData, LightbulbScoreType
from api.analyser import calculate_lightbulb_score

from dao.attribute import DAOAttributePL
from models.attribute import AttributePLInDB

app = FastAPI()
dao_analysis: DAOAnalysis = DAOAnalysis()
dao_document: DAODocument = DAODocument()
dao_attribute: DAOAttributePL = DAOAttributePL(collection_name=API_ATTRIBUTES_COLLECTION_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins if needed
    allow_methods=["*"],  # Adjust this to restrict HTTP methods if needed
    allow_headers=["*"],  # Adjust this to restrict headers if needed
)

def _handle_analysis_status(analysis: AnalysisInDB) -> BackgroundTaskStatusResponse:
    if analysis.status == AnalysisStatus.FAILED:
        return BackgroundTaskFailedResponse(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_id,
            estimated_wait_time=0
        )
    elif analysis.status == AnalysisStatus.RUNNING:
        estimated_end_time = analysis.start_time + timedelta(seconds=analysis.estimated_wait_time)
        remaining_time = (estimated_end_time - datetime.now()).total_seconds()
        if remaining_time < 10:
            # if remaining time is less than 10 seconds, set it to 30 seconds
            remaining_time = 30
        return BackgroundTaskRunningResponse(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_id,
            estimated_wait_time=remaining_time
        )
    elif analysis.status == AnalysisStatus.FINISHED:
        return BackgroundTaskRunningResponse(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_id,
            estimated_wait_time=0
        )
    else:
        raise Exception(f"Unknown analysis status: {analysis.status}")

@app.get("/document-analysis-status",
         response_model=BackgroundTaskStatusResponse | NoAnalysisFoundResponse)
def document_analysis_results(analysis_id: str):
    analysis: Optional[AnalysisInDB] = dao_analysis.find_one_by_query({'analysis_id': analysis_id})
    if not analysis:
        return NoAnalysisFoundResponse()

    return _handle_analysis_status(analysis)


@app.get("/document-analysis-result",
         response_model=AnalysisResultsResponse | BackgroundTaskStatusResponse | NoAnalysisFoundResponse)
def document_analysis_results(analysis_id: str):
    analysis: Optional[AnalysisInDB] = dao_analysis.find_one_by_query({'analysis_id': analysis_id})
    if not analysis:
        return NoAnalysisFoundResponse()

    if analysis.status != AnalysisStatus.FINISHED:
        return _handle_analysis_status(analysis)

    attribute: AttributePLInDB = dao_attribute.find_one_by_query({'_id': analysis.features_id})
    if not attribute:
        return NoAttributeFoundResponse()

    return AnalysisResultsResponse(
        analysis_data=AnalysisData(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_id,
            full_features=attribute
        )
    )


@app.get("/lightbulbs_scores",
         response_model=LightbulbScoreResponse)
def lightbulb_score(analysis_id: str, request_data: LightbulbScoreRequestData):
    analysis: Optional[AnalysisInDB] = dao_analysis.find_one_by_query({'analysis_id': analysis_id})
    if not analysis:
        return NoAnalysisFoundResponse()

    if analysis.status != AnalysisStatus.FINISHED:
        return _handle_analysis_status(analysis)

    attribute: AttributePLInDB = dao_attribute.find_one_by_query({'_id': analysis.features_id})
    if not attribute:
        return NoAttributeFoundResponse()

    attribute_dict = attribute.to_flat_dict_normalized()
    lightbulb_score_data = []
    for attribute_name in request_data.attribute_names:
        if attribute_name not in attribute_dict:
            continue

        attribute_value = attribute_dict[attribute_name]#FIXME check what category should the score be in
        lightbulb_score_value = calculate_lightbulb_score(attribute_value, attribute_name, category=LightbulbScoreType.BIDIRECTIONAL)
        lightbulb_score_data.append(LightbulbScoreData(
            attribute_name=attribute_name,
            type=LightbulbScoreType.BIDIRECTIONAL,
            score=lightbulb_score_value
        ))

    return LightbulbScoreResponse(
        lightbulb_scores=lightbulb_score_data
    )
    # return list of lightbulb scores

@app.get("/graph-image")
def get_graph_image(analysis_id: str, attribute_name: str):
    # generate image and save it to a file
    image_path = "path/to/image.png"
    return FileResponse(image_path, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)