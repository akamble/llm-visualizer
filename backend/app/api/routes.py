"""HTTP endpoints. Thin layer: parse request -> call core function -> return.

Adding a new lesson later means adding one route here and one module in
`app/core/`. Nothing else needs to change.
"""

from fastapi import APIRouter

from app.core import concepts
from app.core.text_pipeline import run_pipeline
from app.core.tokenization import run_tokenization
from app.models.schemas import (
    PipelineResponse,
    TextRequest,
    TokenizationRequest,
    TokenizationResponse,
)

router = APIRouter(prefix="/api", tags=["lessons"])


@router.get("/concepts/hierarchy")
def concepts_hierarchy():
    """The AI ⊃ ML ⊃ DL ⊃ LLM big-picture diagram data."""
    return concepts.get_hierarchy()


@router.get("/concepts/roadmap")
def concepts_roadmap():
    """The basic -> advanced -> agentic learning path."""
    return {"modules": concepts.get_roadmap()}


@router.post("/text/pipeline", response_model=PipelineResponse)
def text_pipeline(req: TextRequest):
    """Lesson 1: run text through normalize -> tokenize -> vectors -> hidden layer."""
    return run_pipeline(req.text)


@router.post("/text/tokenize", response_model=TokenizationResponse)
def text_tokenize(req: TokenizationRequest):
    """Lesson 2: train BPE merges, then encode the text into subword tokens."""
    return run_tokenization(req.text, req.num_merges)
