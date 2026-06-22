from typing import List, Dict, Union, Any
from span_aligner import SpanProjector


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
        
        formatted_span = {**span}
        formatted_span.update({
            "start": span["start"],
            "end": span["end"],
            "labels": labels,
            "text": src_text[span["start"]:span["end"]]
        })
        formatted_spans.append(formatted_span)

    # Use the projector to project spans from source to target text
    projector = SpanProjector()
    projected_spans = projector.project_spans(src_text, tgt_text, formatted_spans, max_gap=max_gap)

    formatted_projected_spans = []
    for span in projected_spans:
        formatted_span = {**span}  # keep all projected fields

        formatted_span.update({
            "start": span["start"],
            "end": span["end"],
            "label": span.get("labels", [None])[0] if span.get("labels", []) else None,
            "text": tgt_text[span["start"]:span["end"]]
        })

        formatted_projected_spans.append(formatted_span)

    return formatted_projected_spans