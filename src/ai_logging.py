"""Helpers for recording AI model calls via Task.record_ai_call."""
import re
import time
from typing import Optional

_AIRO_PREFIX = "http://data.lblod.info/ontology/airo#"


def model_name_to_uri(model_name: str) -> str:
    """Convert a model name/identifier to an airo: model URI.

    Examples:
        "spacy-en-core" -> "http://data.lblod.info/ontology/airo#spacy-en-core"
        "Helsinki-NLP/opus-mt-nl-en" -> "http://data.lblod.info/ontology/airo#helsinki-nlp-opus-mt-nl-en"
    """
    safe = re.sub(r'[^A-Za-z0-9]+', '-', model_name).strip('-').lower()
    return f"{_AIRO_PREFIX}{safe}"


def _resolve_model_uri(model_uri: str) -> str:
    """Resolve a model identifier to a proper URI.

    If *model_uri* is already an absolute URI (starts with ``http``), it is
    returned unchanged.  Otherwise it is treated as a model name and passed
    through :func:`model_name_to_uri`.
    """
    if model_uri.startswith("http"):
        return model_uri
    return model_name_to_uri(model_uri)


def extract_tokens_from_response(response) -> tuple[int, int]:
    """Extract input/output token counts from a LangChain chat response.

    Returns (tokens_in, tokens_out). Falls back to (0, 0) if unavailable.
    """
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return (
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
    return (0, 0)


def record_llm_call(
    task,
    endpoint: str,
    model_uri: str,
    response,
    duration: float,
):
    """Record an LLM API call on *task* using usage metadata from *response*.

    Args:
        task: A Task instance with record_ai_call().
        endpoint: Base URL or provider identifier.
        model_uri: Absolute airo URI or model name/identifier.  If already an
            absolute URI it is used as‑is; otherwise it is converted via
            :func:`model_name_to_uri`.
        response: LangChain chat response (for token extraction).
        duration: Elapsed seconds for the call.
    """
    tokens_in, tokens_out = extract_tokens_from_response(response)
    task.record_ai_call(
        endpoint=endpoint,
        model_uri=_resolve_model_uri(model_uri),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        duration=duration,
    )


def record_ml_call(
    task,
    endpoint: str,
    model_uri: str,
    duration: float,
):
    """Record a classic ML model inference on *task* (tokens=0, cost not computed).

    Args:
        task: A Task instance with record_ai_call().
        endpoint: Base URL or "local".
        model_uri: Absolute airo URI or model name/identifier.  If already an
            absolute URI it is used as‑is; otherwise it is converted via
            :func:`model_name_to_uri`.
        duration: Elapsed seconds for the call.
    """
    task.record_ai_call(
        endpoint=endpoint,
        model_uri=_resolve_model_uri(model_uri),
        tokens_in=0,
        tokens_out=0,
        duration=duration,
    )
