from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.analyser import load_reference_attributes

from api.analysis_fetcher import router as analysis_fetcher_router
from api.feature_extraction import router as feature_extraction_router
from api.db_calls import router as db_router
from api.server_config import ANALYSIS_TASK_QUEUE

from config import init_all_polish_models
from services.utils import suppress_stdout

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading reference attributes...")
    load_reference_attributes()
    print("Reference attributes loaded successfully.")
    print("=========================================================")

    print("Initializing feature extraction models...")
    with suppress_stdout():
        init_all_polish_models()
    print("Feature extraction models initialized successfully.")
    print("=========================================================")

    print("Starting the analysis queue worker...")
    ANALYSIS_TASK_QUEUE.start_worker()
    print("Analysis queue worker started successfully.")
    print("=========================================================")

    yield
app = FastAPI(lifespan=lifespan)

# add the routers
app.include_router(analysis_fetcher_router, prefix="/results", tags=["Results"])
app.include_router(feature_extraction_router, prefix="/analysis", tags=["Feature Extraction"])
app.include_router(db_router, prefix="/db-operations", tags=["Database Operations"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins if needed
    allow_methods=["*"],  # Adjust this to restrict HTTP methods if needed
    allow_headers=["*"],  # Adjust this to restrict headers if needed
)

# entry point when you run:  python main.py
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.server:app",       # points to this file and the FastAPI instance
        host="0.0.0.0",
        port=8989,
        reload=True,      # auto reload during development
    )