from typing import List, Dict, Union, Any
from span_aligner import SpanProjector

# Singleton instance
_projector = None

def _get_projector() -> SpanProjector:
    """Lazy load the SpanProjector singleton."""
    global _projector
    if _projector is None:
        _projector = SpanProjector()
    return _projector

def project_spans(
    src_text: str, 
    tgt_text: str, 
    src_spans: Union[List, Dict],
    max_gap: int = 5
) -> List[Dict]:
    """Projects Entities and Segments from source text to target text using the singleton SpanProjector."""
    
    # 1. Normalize input to always be a list
    if isinstance(src_spans, dict):
        src_spans = src_spans.get("spans", [])
    
    if not src_spans:
        return []

    formatted_spans = []

    # 2. Process spans
    for span in src_spans:
        # specific logic: prefer single 'label', fallback to 'labels' list
        labels = [span["label"]] if "label" in span else span.get("labels", [])
        
        formatted_spans.append({
            "start": span["start"],
            "end": span["end"],
            "labels": labels,
            "text": src_text[span["start"]:span["end"]]
        })

    # Use the projector to project spans from source to target text
    projected_spans = _get_projector().project_spans(src_text, tgt_text, formatted_spans, max_gap=max_gap)

    formatted_projected_spans = []
    for span in projected_spans:
        formatted_projected_spans.append({
            "start": span["start"],
            "end": span["end"],
            "label": span.get("labels", [None])[0] if span.get("labels", []) else None,
            "text": tgt_text[span["start"]:span["end"]]
        })

    return formatted_projected_spans