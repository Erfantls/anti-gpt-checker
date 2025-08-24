from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.feature_extraction import router as feature_extraction_router, init_analysis_executor
from api.server_config import ANALYSIS_TASK_QUEUE, init_analysis_task_queue

from config import init_all_polish_models
from services.utils import suppress_stdout

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing feature extraction models...")
    with suppress_stdout():
        init_all_polish_models()
    print("Feature extraction models initialized successfully.")
    print("=========================================================")

    print("Starting the analysis queue worker...")
    init_analysis_task_queue()
    ANALYSIS_TASK_QUEUE.start_worker()
    init_analysis_executor()
    print("Analysis queue worker started successfully.")
    print("=========================================================")

    yield
app = FastAPI(lifespan=lifespan)

# add the routers
app.include_router(feature_extraction_router, prefix="/analysis", tags=["Feature Extraction"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins if needed
    allow_methods=["*"],  # Adjust this to restrict HTTP methods if needed
    allow_headers=["*"],  # Adjust this to restrict headers if needed
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.server_analysis:app",       # points to this file and the FastAPI instance
        host="0.0.0.0",
        port=8990,
        reload=True,      # auto reload during development
    )