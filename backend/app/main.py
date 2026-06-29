"""FastAPI application entry point.

Run locally with:
    uvicorn app.main:app --reload --port 8000

Then open http://localhost:8000/docs to try the API interactively.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="LLM Visualizer API",
    description="Learn AI from basics to Agentic AI — one visual lesson at a time.",
    version="0.1.0",
)

# Allow the Vite dev server (and any localhost port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    """Health check / friendly landing message."""
    return {
        "message": "LLM Visualizer backend is running.",
        "docs": "/docs",
        "lessons": [
            "/api/concepts/hierarchy",
            "/api/concepts/roadmap",
            "/api/text/pipeline",
            "/api/text/tokenize",
        ],
    }
