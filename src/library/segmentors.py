import re
import asyncio
from abc import ABC, abstractmethod
from span_aligner import SpanAligner
from typing import Optional, Any, Type, List, Dict
from helpers import logger
from pydantic import BaseModel
try:
    from transformers import pipeline as transformers_pipeline
except ImportError:
    # Allow module import even if transformers is missing (e.g. for LLMSegmentor)
    transformers_pipeline = None

# Try importing LLMAnalyzer
try:
    from .LLMAnalyzer import LLMAnalyzer
except ImportError:
    logger.warning("LLMAnalyzer not found. LLMSegmentor will not utilize LLM capabilities.")
    LLMAnalyzer = None


#----------------------------------------------
# Segmentor interface
#----------------------------------------------

class AbstractSegmentor(ABC):
    """Abstract base class for a segmentation strategy."""
    
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None, temperature: float = 0.1, max_new_tokens: int = 2000):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
    
    @abstractmethod
    def segment(self, text: str) -> List[Dict[str, Any]]:
        """
        Segments text into labeled spans.
        Returns list of dicts: {'label': str, 'start': int, 'end': int, 'text': str}
        """
        pass


#----------------------------------------------
# Segmentor implementations
#---------------------------------------------

# Legacy Gemma-based segmentor with tag repair and alignment logic
class GemmaSegmentor(AbstractSegmentor):
    """
    Common logic for models that output XML-like tags (e.g. <TITLE>text</TITLE>).
    Provides methods to repair tags and align generated spans to source text.
    """
    _generator = None
    _TAG_RE = re.compile(r"</?([A-Za-z0-9_]+)>")

    SYSTEM_INSTRUCTION = """Your task is to segment the given text by inserting XML-style boundary tags. For each requested segment type, you must wrap the corresponding part of the text with a start tag <SEGMENT_NAME> and an end tag </SEGMENT_NAME>.

Rules:
- Do not modify or alter any text from the decision.
- Insert the tags directly into the original text at the correct boundaries.
- If a segment is not present in the text, do not add tags for that segment.
- For the segment 'ARTICLE', each article must be wrapped separately using <ARTICLE>...</ARTICLE>, one article per tag pair.
- Only use segment names from the provided list. Do not invent or use any other segment names.
- The output must contain the original text with the correct tags inserted.

SEGMENTS:
```
['TITLE', 'PARTICIPANTS', 'MOTIVATION', 'PREVIOUS_DECISIONS', 'LEGAL_FRAMEWORK', 'DECISION', 'VOTING', 'ARTICLE']
```"""
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = "wdmuer/decide-marked-segmentation", temperature: float = 0.1, max_new_tokens: int = 4096):
        super().__init__(api_key, base_url, model_name, temperature, max_new_tokens)
   

    def get_generator(self):
        """Lazy-load the segmentation model using config settings."""
        if self.__class__._generator is None:
            if transformers_pipeline is None:
                raise ImportError("transformers library or pipeline not available")
            
            logger.info(f"Loading segmentation model: {self.model_name}")
            
            self.__class__._generator = transformers_pipeline(
                "text-generation",
                model=self.model_name,
            )
            logger.info(f"Loaded segmentation model: {self.model_name}")
        return self.__class__._generator

    def _rstrip_with_tail(self, s: str) -> tuple[str, str]:
        """Return (head_without_trailing_ws, trailing_ws)."""
        m = re.search(r"\s*\Z", s)
        if not m:
            return s, ""
        return s[:m.start()], s[m.start():]

    def _lstrip_with_head(self, s: str) -> tuple[str, str]:
        """Return (leading_ws, tail_without_leading_ws)."""
        m = re.match(r"\A\s*", s)
        if not m:
            return "", s
        return s[:m.end()], s[m.end():]

    def fix_missing_tags(self, text: str, separator: str = "\n\n") -> str:
        """Best-effort repair for malformed tag sequences in the model output.

        The model sometimes emits unmatched or mis-nested tags. This function tries to
        produce a valid tag stream so that we can reliably extract spans afterwards.
        """
        stack: list[str] = []
        out: list[str] = []

        pos = 0
        for m in self._TAG_RE.finditer(text):
            chunk = text[pos:m.start()]
            tag = m.group(0)
            name = m.group(1)
            is_closing = tag.startswith("</")

            if not is_closing:
                # OPENING TAG
                if stack:
                    head, tail_ws = self._rstrip_with_tail(chunk)
                    out.append(head)
                    while stack:
                        out.append(f"</{stack.pop()}>")
                    if tail_ws:
                        out.append(tail_ws)
                    out.append(tag)
                else:
                    out.append(chunk)
                    out.append(tag)
                stack.append(name)
            else:
                # CLOSING TAG
                if stack and stack[-1] == name:
                    out.append(chunk)
                    out.append(tag)
                    stack.pop()
                elif name not in stack:
                    lead_ws, body = self._lstrip_with_head(chunk)
                    out.append(lead_ws)
                    out.append(f"<{name}>")
                    out.append(body)
                    out.append(tag)
                else:
                    head, tail_ws = self._rstrip_with_tail(chunk)
                    out.append(head)
                    while stack and stack[-1] != name:
                        out.append(f"</{stack.pop()}>")
                    out.append(tag)
                    stack.pop()
                    out.append(tail_ws)
            pos = m.end()

        out.append(text[pos:])

        if stack:
            all_tail = out.pop()
            head, tail_ws = self._rstrip_with_tail(all_tail)
            out.append(head)
            while stack:
                out.append(f"</{stack.pop()}>")
            out.append(tail_ws)

        return "".join(out)

    def extract_entities_ner_style(self, tagged_text: str) -> tuple[str, list[dict[str, Any]]]:
        """Extract spans from the XML-tagged text.

        The returned offsets are relative to `clean_text` (tagged_text with tags removed).
        """
        TAG_PATTERN = re.compile(r"<(/?)([A-Za-z0-9_]+)>")
        
        entities = []
        clean_parts = []
        clean_pos = 0
        
        open_tags: list[tuple[str, int]] = []
        
        last_end = 0
        for match in TAG_PATTERN.finditer(tagged_text):
            # Add text before this tag to clean output
            text_before = tagged_text[last_end:match.start()]
            clean_parts.append(text_before)
            clean_pos += len(text_before)
            
            is_closing = match.group(1) == "/"
            tag_name = match.group(2)
            
            if not is_closing:
                # Opening tag - record start position
                open_tags.append((tag_name, clean_pos))
            else:
                # Closing tag - find matching open tag and create entity
                for i in range(len(open_tags) - 1, -1, -1):
                    if open_tags[i][0] == tag_name:
                        start_pos = open_tags[i][1]
                        entities.append({
                            "start": start_pos,
                            "end": clean_pos,
                            "label": tag_name,
                            "_temp_end": clean_pos
                        })
                        open_tags.pop(i)
                        break
            
            last_end = match.end()
        
        # Add remaining text after last tag
        clean_parts.append(tagged_text[last_end:])
        clean_text = "".join(clean_parts)
        
        # Populate the 'text' field for each entity
        for entity in entities:
            entity["text"] = clean_text[entity["start"]:entity["end"]]
            del entity["_temp_end"]
        
        # Sort entities by start position
        entities.sort(key=lambda x: x["start"])
        
        return clean_text, entities

    def align_segments(self, source_text: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aligns raw segments (from generated/cleaned text) to the original source text.
        """
        aligned_segments = []
        for seg in segments:
            text = seg.get('text', '').strip()
            if len(text) < 3:
                continue
            
            pos = source_text.find(text)
            if pos >= 0:
                aligned_segments.append({
                    'label': seg['label'],
                    'start': pos,
                    'end': pos + len(text),
                    'text': text
                })
            else:
                logger.warning(f"Could not find '{seg['label']}' in source: {text[:50]}...")
        
        logger.info(f"Aligned {len(aligned_segments)}/{len(segments)} segments to source")
        return aligned_segments


    # Segmentation method
    def segment(self, text: str) -> List[Dict[str, Any]]:
        logger.info("Running Gemma segmentation...")
        
        messages = [
            {"role": "system", "content": self.SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"PUBLIC DECISION:\n```\n{text}\n```"},
        ]
        
        generator = self.get_generator()
        output = generator(
            messages,
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
            temperature=self.temperature,
            top_p=0.95,
            do_sample=True,
        )
        
        raw_output = output[0]["generated_text"]
        fixed_output = self.fix_missing_tags(raw_output)
        _, segments = self.extract_entities_ner_style(fixed_output)
        
        # Align derived segments to original source
        return self.align_segments(text, segments)

class SegmentationSchemaModel(BaseModel):
    tagged_text: str = ""
    document_classification: str = ""


# Generic LLM-based segmentor that relies on the LLM to produce JSON output with tagged text, then extracts spans using the span_aligner package.
class LLMSegmentor(AbstractSegmentor):
    """
    Segmentor using LLMAnalyzer.
    """

    LABEL_MAPPING = {
    "decision_title": "TITLE",
    "participants" : "PARTICIPANTS",
    "motivation": "MOTIVATION",
    "previous_decisions": "PREVIOUS DECISIONS",
    "legal_framework": "LEGAL FRAMEWORK",
    "decision": "DECISION",
    "article": "ARTICLE",
    "voting": "VOTING",
    "administrative_body": "ADMINISTRATIVE BODY",
    "publication_date": "PUBLICATION DATE"
    }

    SYSTEM_PROMPT_REFERENCES_SEGMENTATION = """
    You are a precise document-segmentation and classification assistant for
    decisions of local governments and municipalities.

    You do NOT reproduce the document. You output only the CLASSIFICATION and a
    list of tag SPANS expressed as line-number ranges. A separate deterministic
    program inserts the tags into the original text using your spans, so you must
    never emit, rewrite, paraphrase, normalize, or reorder document text.

    INPUT FORMAT
    - The document is given with every line prefixed by a marker: "LNNNN| ".
    - These markers are metadata, NOT document content. Reference the numbers only.
    - Line numbering is contiguous and includes blank lines.

    OUTPUT CONTRACT
    - Respond with STRICT JSON only. No markdown fences, no commentary, no extra keys.
    - Exactly two keys: "document_classification" and "spans".
    - "spans" is a (possibly empty) array of objects:
        { "tag": <one allowed tag>,
          "start_line": <int, the LNNNN where the span begins, inclusive>,
          "end_line":   <int, the LNNNN where the span ends,   inclusive>,
          "text":       <optional exact substring; see SUB-LINE rule> }

    ALLOWED TAGS (only these):
    The "tag" value is the BARE tag name with NO angle brackets.
    Allowed values: document_title, decision_title, decision_outcome,
    administrative_body, publication_date, participants, motivation,
    previous_decisions, legal_framework, decision, article, voting,
    attachments, attachment

    SPAN & NESTING RULES
    - Express nesting implicitly by range containment. If span A's [start,end] is
      inside span B's [start,end], A is nested in B. Do NOT create overlaps that are
      not clean containment.
    - When two spans share the SAME start_line and end_line but one contains the
      other (e.g. <decision> containing <decision_outcome>), list the CONTAINER
      first in the array.
    - <legal_framework> and <previous_decisions> must fall inside a <motivation>
      span. Start <motivation> early enough to enclose leading legal citations.
    - <article>, <voting>, and <decision_outcome> must fall inside a <decision>
      span. If voting appears before the articles, start <decision> early enough to
      enclose it.
    - <decision_title> is ALWAYS top-level. It must NEVER be inside a <decision>
      span. Open the <decision> span only after the <decision_title> line(s).
    - <administrative_body> is ALWAYS top-level; never inside <decision>.
    - Each <attachment> must fall inside the <attachments> span. Attachment content
      belongs to no other tag.
    - Content mapping:
        * "Meeting presentation" / "Summary" -> inside <motivation>.
        * "Resolution" / "Meeting resolution" / "It is decided" -> inside <decision>.
        * "(majority approved)", "(approved with one dissenting vote)", tallies,
          roll calls, "unanimously" -> inside <voting>.
        * "Financial impact" -> inside <attachments>.
    - DECISION BOUNDARY: the <decision> span ends after voting and all articles. Do
      NOT extend it over signature/certification blocks (names, titles, signatures).
    - Omit any tag whose content is absent. Never produce empty or zero-width spans.

    SUB-LINE RULE (use sparingly)
    - If a span covers only PART of a single line (e.g. <decision_outcome> should
      cover only the leading status word "Approval" while "subject to conditions"
      stays untagged), set start_line == end_line and provide "text" = the exact
      literal substring to tag, copied verbatim from that line (excluding the
      "LNNNN| " marker). The program wraps the first occurrence on that line.
    - Otherwise omit "text"; the span covers the whole referenced line range.

    CLASSIFICATION CATEGORIES
    - "Minute": one full decision with all main sections (title, motivation,
      decision body, articles, voting).
    - "Minutes": multiple full decisions, each with its own main sections.
    - "Agendapoints": a list of agenda items / decision titles, LACKING motivations,
      articles, voting.
    - "Decision-List": a summary of decision titles plus one-line outcomes
      (dispositions), typically with a participants section but LACKING full
      motivations or detailed articles.
    - "Non-Decision": contains no formal municipal decision text.

    WHAT TO TAG PER CLASSIFICATION
    - "Non-Decision": return "spans": [] and the classification.
    - "Agendapoints": tag <document_title>, <administrative_body>, <publication_date>,
      <participants> where present, and wrap EVERY agenda item line in its own
      top-level <decision_title>. Do not open <decision>.
    - "Decision-List": tag <document_title>, <administrative_body>,
      <publication_date>, <participants> where present, then per item use:
        <decision_title> (top-level), followed by a <decision> span containing a
        <decision_outcome> span (the one-line disposition), OR just <decision_title>
        if no outcome line exists.
      The <decision_title> is NEVER inside the <decision> span.
    - "Minute" / "Minutes": tag all applicable sections using the full tag set and
      all nesting rules above.

    Reason step by step internally, but output ONLY the final JSON.
    """

    USER_PROMPT_TEMPLATE_REFERENCES_SEGMENTATION = """
    STEP 1 — Read the entire line-numbered municipal text below.

    STEP 2 — CLASSIFY it as exactly one of:
    "Minute", "Minutes", "Agendapoints", "Decision-List", "Non-Decision"
    (definitions are in the system instructions).

    STEP 3 — Produce the SPANS:
    - Reference lines by their LNNNN numbers only. Do not reproduce document text
      except in an optional "text" anchor for a sub-line span.
    - Follow all span, nesting, boundary, and per-classification rules from the
      system instructions.
    - Omit tags whose content is absent. No empty or overlapping (non-nested) spans.

    Return STRICT JSON, no fences, exactly:
    {{
      "document_classification": "<one of the five categories>",
      "spans": [
        {{ "tag": "<allowed tag>", "start_line": <int>, "end_line": <int> }}
      ]
    }}

    Line-numbered text:
    {numbered_text}
    """

    RESULTS_SCHEMA_SEGMENTATION = {
        "document_classification": {"default": "", "type": str},
        "spans": {"default": [], "type": list},
        "tagged_text": {"default": "", "type": str}
    }

    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = "mistral-nemo", temperature: float = 0.0, max_new_tokens: int = 14000, provider: str = "ollama", max_retries: int = 3, retry_delay: float = 15.0):
        super().__init__(api_key, base_url, model_name, temperature, max_new_tokens)
        if LLMAnalyzer is None:
             raise ImportError("LLMAnalyzer class is not available.")

        self.analyzer = LLMAnalyzer(
            provider=provider,
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def format_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        
        return {
            "label": segment.get("labels", [])[0] if segment.get("labels") else "UNKNOWN",
            "start": segment.get("start", 0),
            "end": segment.get("end", 0),
            "text": segment.get("text", "")
        }

    def segment(self, text: str) -> List[Dict[str, Any]]:
        logger.info(f"Running LLM segmentation with {self.analyzer.model_name}...")

        try:
            result = self.analyzer.analyze_single_entry(
                text=text,
                system_prompt=self.SYSTEM_PROMPT_REFERENCES_SEGMENTATION,
                user_prompt_template=self.USER_PROMPT_TEMPLATE_REFERENCES_SEGMENTATION,
                expected_schema=self.RESULTS_SCHEMA_SEGMENTATION,
                text_limit=28000,
                preprocess=True,
                postprocess=True,
            )
        except Exception as e:
            logger.exception("LLM segmentation failed")
            raise RuntimeError(
                f"LLM segmentation failed ({self.analyzer.model_name}): {e}"
            ) from e

        tagged_text = result.get("tagged_text", "")
        if not tagged_text:
            logger.warning("LLM returned empty tagged text.")
            return []

        # Project the tags onto the original text
        mapped_tagged_text = SpanAligner.map_tags_to_original(
            original_text=text,
            tagged_text=tagged_text,
            min_ratio=0.7,
            max_dist=200,
        )
        # Fix tags just in case, though LLM should be better
        annotations = SpanAligner.get_annotations_from_tagged_text(
            mapped_tagged_text,
            span_map=self.LABEL_MAPPING
        )
        return [self.format_segment(span) for span in annotations.get("spans", [])]

