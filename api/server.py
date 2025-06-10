from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.analyser import load_reference_attributes
from api.analysis_fetcher import router as analysis_fetcher_router
from api.feature_extraction import router as feature_extraction_router

app = FastAPI()

# add the routers, choose prefixes if you wish
app.include_router(analysis_fetcher_router, prefix="/results")
app.include_router(feature_extraction_router, prefix="/analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins if needed
    allow_methods=["*"],  # Adjust this to restrict HTTP methods if needed
    allow_headers=["*"],  # Adjust this to restrict headers if needed
)

# entry point when you run:  python main.py
if __name__ == "__main__":
    load_reference_attributes()

    import uvicorn

    uvicorn.run(
        "api.server:app",       # points to this file and the FastAPI instance
        host="0.0.0.0",
        port=8989,
        reload=True,      # auto reload during development
    )