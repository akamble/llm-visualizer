"""Request/response shapes shared across the API (pydantic models).

Kept deliberately loose (`dict`/`Any`) for the teaching endpoints because the
pipeline returns rich, nested, evolving structures and we want the lesson code
to stay readable rather than fighting the type system. Tighten these later as
the data shapes stabilise.
"""

from typing import Any, Dict

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """Input for the text pipeline."""
    text: str = Field(
        default="I love AI",
        description="The sentence to push through the pipeline.",
        examples=["I love AI", "machines learn from data"],
    )


class PipelineResponse(BaseModel):
    """Output of /api/text/pipeline — every intermediate value, for visualizing."""
    input: str
    steps: Dict[str, Any]
    config: Dict[str, Any]
    explainers: Dict[str, str]      # "how is each value computed?"
    field_map: list[Dict[str, str]]  # which AI field each step belongs to
    response: str


class TokenizationRequest(BaseModel):
    """Input for the BPE tokenization lesson."""
    text: str = Field(
        default="slowest learning",
        description="Text to tokenize with the learned BPE merges.",
        examples=["slowest learning", "fastest learner"],
    )
    num_merges: int = Field(
        default=12,
        ge=1,
        le=30,
        description="How many BPE merge rules to learn (drives the slider).",
    )


class TokenizationResponse(BaseModel):
    """Output of /api/text/tokenize — training history + encoding, for visualizing."""
    input: str
    num_merges: int
    corpus: Dict[str, int]
    training: Dict[str, Any]
    encoding: Dict[str, Any]
    comparison: Dict[str, Any]
    explainers: Dict[str, str]
    field_map: list[Dict[str, str]]
    code_sample: str
    response: str
