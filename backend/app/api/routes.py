"""HTTP endpoints. Thin layer: parse request -> call core function -> return.

Adding a new module later means adding one route here and one file in
`app/core/`. Nothing else needs to change.
"""

from fastapi import APIRouter

from app.core import concepts
from app.core.embeddings import run_embeddings
from app.core.positional import run_positional
from app.core.text_pipeline import run_pipeline
from app.core.tokenization import run_tokenization
from app.models.schemas import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    PipelineResponse,
    PositionalRequest,
    PositionalResponse,
    TextRequest,
    TokenizationRequest,
    TokenizationResponse,
)

router = APIRouter(prefix="/api", tags=["modules"])


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
    """Module 1: run text through normalize -> tokenize -> vectors -> hidden layer."""
    return run_pipeline(req.text)


@router.post("/text/tokenize", response_model=TokenizationResponse)
def text_tokenize(req: TokenizationRequest):
    """Module 2: train BPE merges, then encode the text into subword tokens."""
    return run_tokenization(req.text, req.num_merges)


@router.post("/text/embeddings", response_model=EmbeddingsResponse)
def text_embeddings(req: EmbeddingsRequest):
    """Module 3: TF-IDF vectors + cosine similarity + query ranking."""
    return run_embeddings(req.documents, req.query)


@router.post("/text/positional", response_model=PositionalResponse)
def text_positional(req: PositionalRequest):
    """Module 4: sinusoidal positional encoding added to token embeddings."""
    return run_positional(req.text, req.dim)
