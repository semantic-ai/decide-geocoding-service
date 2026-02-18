import logging
import re
import asyncio
from abc import ABC, abstractmethod
from span_aligner import SpanAligner
from typing import Optional, Any, Type, List, Dict
try:
    from transformers import pipeline as transformers_pipeline
except ImportError:
    # Allow module import even if transformers is missing (e.g. for LLMSegmentor)
    transformers_pipeline = None

# Try importing LLMAnalyzer
try:
    from .LLMAnalyzer import LLMAnalyzer
except ImportError:
    logging.getLogger("Segmentation").warning("LLMAnalyzer not found. LLMSegmentor will not utilize LLM capabilities.")
    LLMAnalyzer = None


#----------------------------------------------
# Segmentor interface
#----------------------------------------------

class AbstractSegmentor(ABC):
    """Abstract base class for a segmentation strategy."""
    
    def __init__(self, api_key: str = None, endpoint: str = None, model_name: str = None, temperature: float = 0.1, max_new_tokens: int = 2000):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = api_key
        self.endpoint = endpoint
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
    def __init__(self, api_key: str = None, endpoint: str = None, model_name: str = "wdmuer/decide-marked-segmentation", temperature: float = 0.1, max_new_tokens: int = 4096):
        super().__init__(api_key, endpoint, model_name, temperature, max_new_tokens)
   

    def get_generator(self):
        """Lazy-load the segmentation model using config settings."""
        if self.__class__._generator is None:
            if transformers_pipeline is None:
                raise ImportError("transformers library or pipeline not available")
            
            self.logger.info(f"Loading segmentation model: {self.model_name}")
            
            self.__class__._generator = transformers_pipeline(
                "text-generation",
                model=self.model_name,
            )
            self.logger.info(f"Loaded segmentation model: {self.model_name}")
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
                self.logger.warning(f"Could not find '{seg['label']}' in source: {text[:50]}...")
        
        self.logger.info(f"Aligned {len(aligned_segments)}/{len(segments)} segments to source")
        return aligned_segments


    # Segmentation method
    def segment(self, text: str) -> List[Dict[str, Any]]:
        self.logger.info("Running Gemma segmentation...")
        
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


# Generic LLM-based segmentor that relies on the LLM to produce JSON output with tagged text, then extracts spans using the span_aligner package.
class LLMSegmentor(AbstractSegmentor):
    """
    Segmentor using LLMAnalyzer (Azure OpenAI / Local LLM).
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
    You are a precise document-segmentation and tagging assistant.
    You must respond with valid JSON only.

    Core invariants (critical):
    - Preserve every character of the input text exactly (including whitespace, line breaks, punctuation, numbering, and layout).
    - Do NOT delete, paraphrase, reorder, summarize, or normalize text.
    - The ONLY modification allowed is inserting the specified tags inline at appropriate boundaries.

    Allowed tags (and only these): <document_title>, <decision_title>, <decision_outcome>, <publication_date>, <participants>, <motivation>, <previous_decisions>, <legal_framework>, <decision>, <article>, <voting>, <attachments>, <attachment>.

    Segmentation and nesting rules:
    GLOBAL INVARIANTS (apply to ALL classifications):
    - <decision_title> is ALWAYS a top-level span; it MUST NEVER appear inside <decision>.
    - Open <decision> ONLY after closing the related </decision_title> (never before or around it).
    - For Decision-List: <decision> will normally contain ONLY <decision_outcome>.
    - Document Title (<document_title>): wrap the complete formal agenda/document heading (often includes meeting type/location/date).
    - Decision Title (<decision_title>): wrap the formal, specific decision heading describing the action/topic.
    - Decision Outcome (<decision_outcome>): ONLY for Decision-List or Agenda summary formats: one-line disposition/status following a <decision_title> (e.g., "Approval", "Conditional approval", "Refusal", "Acknowledgement", "Accepted"). Must be the IMMEDIATE next non-empty line; exclude any longer explanatory text.
    - Always wrap <decision_outcome> inside a <decision> block if not already present.
    - Participants & Attendance (<participants>): wrap the entire attendance/roles block including subheadings (e.g., "Present", "Excused", "Secretary").
    - Motivation (<motivation>): wrap contextual background, reasoning, justification, including any "introduction" paragraphs.
    - ALWAYS include <legal_framework> and <previous_decisions> inside <motivation> when present.
    - If legal citations appear before the main motivation text, start <motivation> earlier to encompass them.
    - Sections like "Meeting presentation" or "Summary" belong to the <motivation> block.
    - Previous Decisions (<previous_decisions>): resolutions explicitly referenced (e.g., earlier council decisions). Always nest inside <motivation>.
    - Legal Framework (<legal_framework>): citations of laws/decrees/regulations ("Whereas", references to statutory articles). Always nest inside <motivation>.
    - Decision (<decision>): operative part beginning at headings like "Decision", "It is decided", "Resolution".
    - NEVER include the <decision_title> inside <decision> (repeat: titles are always outside).
    - Article (<article>): within <decision>, wrap each discrete provision labelled "Article", "Art.", numbered item, paragraph sign (§), or clear logical clause.
    - Voting (<voting>): ALWAYS nest inside <decision>. If voting appears before articles, start <decision> earlier to encompass it.
    - Mentions like "(majority approved)", "(approved with one dissenting vote)", or similar remarks are part of the <voting> block.
    - BOUNDARY: The <decision> tag closes after voting and all articles. Do NOT include signature/certification blocks inside <decision>.
    - Attachments (<attachments>/<attachment>): wrap full attachments section; each attachment heading plus body until next attachment (or end) becomes one <attachment>.
    - "Financial impact" sections belong to the <attachments> block.
    - No attachment content appears inside any other tag.

    Interpretation guidance (English cues):
    - Document Title (<document_title>): agenda heading lines with meeting context (includes date/location).
    - Decision Title (<decision_title>): concise formal intent/action.
    - Decision Outcome (<decision_outcome>): single disposition/status token (approval/refusal/acknowledgement/conditional approval). If extra commentary appears on same line (e.g., "Approval subject to conditions"), tag only the leading status word/phrase and leave commentary untagged.
    - Publication Date (<publication_date>): date of publication/report (preserve formatting; omit if ambiguous).
    - Legal Framework (<legal_framework>): statutory citations, decrees, articles, competence references.
    - Previous Decisions (<previous_decisions>): explicit references to prior resolutions or numbered decisions.
    - Motivation (<motivation>): narrative justification ("considering that", explanatory paragraphs, meeting presentations, summaries).
    - Decision (<decision>): operative directive ("decides", "it is decided that").
    - Article (<article>): clauses beginning with "Article"/"Art."/numbered bullet/§.
    - Voting (<voting>): outcomes like "unanimous", vote tallies, roll calls, and remarks like "majority approved". Typically follows decision body and often marked as "(unanimously)", "(with one dissenting vote)", etc.
    - Participants (<participants>): attendance lists ("Present", "Excused", roles).
    - Attachments (<attachments>/<attachment>): attachment headings "Appendix", etc. including their body isolated from other tags.

    Tagging behavior:
    - Insert opening/closing tags inline around original spans only.
    - Do not add, remove, or alter characters outside tags.
    - Omit tags if their content absent (no empty tags).
    - Never translate or normalize dates, numbers, casing.
    - Attachments contain full attachment bodies inside <attachments>/<attachment> and are excluded from all other tags.
    Output contract:
    - Return strict JSON with exactly one key: "tagged_text" containing original text plus inserted tags.
    - No markdown fences, no commentary, no additional keys.
    """

    USER_PROMPT_TEMPLATE_REFERENCES_SEGMENTATION = """

    STEP-BY-STEP INSTRUCTIONS:

    STEP 1: 
    Read the entire municipal decision text carefully.

    STEP 2: CLASSIFICATION
    Classify the document into one of the following four categories:
    - "Minute": A single document containing one full municipal decision with all main sections (title, motivation, decision body, articles, voting).
    - "Minutes": A document containing multiple full decisions (e.g., meeting minutes), where each decision contains its own main sections (motivation, articles, etc.).
    - "Agendapoints": A document listing agenda items that includes the decision titles that will be discussed, but LACKING decisions, motivations, articles, and voting details.
    - "Decision-List": A summary document containing titles and short descriptions of decisions or one-line outcomes (dispositions) of decisions, typically with a Participants section but LACKING full motivations or detailed articles.
    - "Non-Decision": Does not contain any formal municipal decision text.


    STEP 3: TAGGING
    IF classification is "Non-Decision":
    - Keep the tagged_text empty and set document_classification accordingly.

    IF classification is one of "Agendapoints" or "Decision-List":
    - Tag only the available sections:
    - <document_title>
    - <decision_title>
    - <decision_outcome> (Decision-List only)
    - <administrative_body>
    - <publication_date>
    - <participants>
    - <decision>

    Agendapoints:
        - Wrap EVERY agenda item line in a top-level <decision_title>.
        - Never wrap the <decision_title> inside <decision>.
    Decision-List:
        - Pattern per item (correct): <decision_title>... </decision_title><decision><decision_outcome>...</decision_outcome></decision> or <decision_title>... </decision_title><decision>...</decision>  or just <decision_title>... </decision_title>.
        - INVALID (never output): <decision><decision_title>...<decision_outcome>...</decision_outcome></decision>.

    Examples:
        Correct Decision-List snippet:
        <decision_title>1. Approval of minutes of the meeting of July 18, 2023.</decision_title>
        <decision><decision_outcome>Approved.</decision_outcome></decision>
    
        Invalid nesting (DO NOT produce):
        <decision><decision_title>1. Approval of minutes of the meeting of July 18, 2023.</decision_title><decision_outcome>Approved.</decision_outcome></decision>

    IF classification is one of "Minute" or "Minutes":
    - Tag the following municipal decision text by inserting ONLY the specified tags inline, preserving every character and the original layout exactly. Do not remove, move, or change any text beyond inserting tags.

    Allowed Tags (exactly as listed):
    <document_title>, <decision_title>, <decision_outcome>, <administrative_body>, <publication_date>, <participants>, <motivation>, <previous_decisions>, <legal_framework>, <decision>, <article>, <voting>, <attachments>, <attachment>

    Critical rules:
    1. PRESERVE ALL TEXT: Insert tags only; do not delete, move, reorder, correct or modify any character.
    2. NESTING (mandatory):
    - Do NOT tag <administrative_body> inside <decision> blocks. Only tag it at top-level.
    - <legal_framework> and <previous_decisions> ALWAYS nest inside <motivation>. Start <motivation> early if needed to include them.
    - <voting> ALWAYS nests inside <decision>. Start <decision> early if voting appears before articles.
    - Mentions like "(majority approved)" or "(approved with one dissenting vote)" belong inside <voting>.
    - Sections like "Meeting presentation" or "Summary" belong inside <motivation>.
    - Sections like "Meeting resolution" or "Resolution" belong inside <decision>.
    - "Financial impact" belongs inside <attachments>.
    - <article> tags always nest inside <decision>.
    - <decision_title> NEVER nests inside <decision>. (Applies globally; repeat for emphasis.)
    - <decision_outcome> ALWAYS nests inside the related <decision> block; never top-level.
    - Each <attachment> nests inside <attachments>.
    3. DECISION BOUNDARIES: Close </decision> after voting and all articles. Do NOT include signature/certification blocks (names, titles) inside <decision>.
    4. ATTACHMENTS: Wrap the entire attachments section (headings + bodies) in <attachments>, with each individual attachment wrapped in <attachment>. Exclude attachment content from all other tags.
    5. NO EMPTY TAGS: Omit any tag if its content is not present in the source text.

    Return STRICT JSON as:
    {{
    "tagged_text": "<ORIGINAL TEXT WITH TAGS INSERTED INLINE, PRESERVING ALL CONTENT>",
    "document_classification": "<DOCUMENT CLASSIFICATION RESULT>"
    }}

    Text to tag:
    {text}
    """


    RESULTS_SCHEMA_SEGMENTATION = {
        "tagged_text": {"default": "", "type": str},
        "document_classification": {"default": "", "type": str}
        }

    def __init__(self, api_key: str = None, endpoint: str = None, model_name: str = "gpt-4.1", temperature: float = 0.0, max_new_tokens: int = 14000):
        super().__init__(api_key, endpoint, model_name, temperature, max_new_tokens)
        if LLMAnalyzer is None:
             raise ImportError("LLMAnalyzer class is not available.")
        
        self.analyzer = LLMAnalyzer(
            api_key=self.api_key, 
            endpoint=self.endpoint, 
            deployment=self.model_name
        )

    def format_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        
        return {
            "label": segment.get("labels", [])[0] if segment.get("labels") else "UNKNOWN",
            "start": segment.get("start", 0),
            "end": segment.get("end", 0),
            "text": segment.get("text", "")
        }

    async def async_segment(self, text: str) -> List[Dict[str, Any]]:
        self.logger.info(f"Running LLM segmentation with {self.analyzer.deployment}...")
        
        try:
             result = await self.analyzer.analyze_single_entry(
                text=text,
                system_prompt=self.SYSTEM_PROMPT_REFERENCES_SEGMENTATION,
                user_prompt_template=self.USER_PROMPT_TEMPLATE_REFERENCES_SEGMENTATION,
                expected_schema=self.RESULTS_SCHEMA_SEGMENTATION,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                text_limit=28000
            )
        except Exception as e:
            self.logger.error(f"LLM segmentation failed: {e}")
            return []

        tagged_text = result.get("tagged_text", "")
        if not tagged_text:
            self.logger.warning("LLM returned empty tagged text.")
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

    def segment(self, text: str) -> List[Dict[str, Any]]:
        # Check for existing event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If in a running loop (e.g., Jupyter), run in a separate thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self.async_segment(text))
                return future.result()
        else:
            return asyncio.run(self.async_segment(text))
