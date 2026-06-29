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
