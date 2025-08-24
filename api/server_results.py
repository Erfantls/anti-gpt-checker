from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.analyser import load_reference_attributes, precompile_gaussian_kde

from api.analysis_fetcher import router as analysis_fetcher_router
from api.db_calls import router as db_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading reference attributes...")
    load_reference_attributes()
    print("Reference attributes loaded successfully.")
    print("=========================================================")

    print("Precompile gaussian kde values...")
    precompile_gaussian_kde()
    print("Precompilation of gaussian kde values completed successfully.")
    print("=========================================================")

    yield
app = FastAPI(lifespan=lifespan)

# add the routers
app.include_router(analysis_fetcher_router, prefix="/results", tags=["Results"])
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
        "api.server_results:app",       # points to this file and the FastAPI instance
        host="0.0.0.0",
        port=8989,
        reload=True,      # auto reload during development
    )