import json
import re
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from ..retry import retry_call
import json_repair
from helpers import logger

MARKER_RE = re.compile(r'^L(\d+)\| ?')


class LLMAnalyzer:
    """
    Analyzer that routes LLM calls through LangChain's init_chat_model factory.
    Supports any provider recognised by LangChain (openai, ollama, mistral, …)
    purely through configuration — no if/else branching.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "mistral-nemo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        json_mode: bool = True,
        max_retries: int = 3,
        retry_delay: float = 15.0,
    ):
        self.model_name = model_name
        self._provider = provider
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        kwargs: Dict[str, Any] = {"temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if json_mode:
            if provider == "ollama":
                kwargs["format"] = "json"
            elif provider in {"mistralai", "mistral"}:
                kwargs["response_format"] = {"type": "json_object"}

        self._chat_model = init_chat_model(
            f"{provider}:{model_name}",
            **kwargs,
            timeout=600,
            max_retries=0,
        )

    def preprocess(self, text: str, min_width: int = 4) -> Tuple[str, List[str], int]:
        """
        numbered_text : LNNNN|-prefixed, '\\n'-joined  -> send this to the LLM
        lines         : exact original pieces (split on '\\n') -> keep for reconstruction
        width         : zero-pad width used
        Contract:  '\\n'.join(lines) == text   (byte-exact, always)
        """
        lines = text.split('\n')
        width = max(min_width, len(str(len(lines))))
        numbered = [
            f"L{i:0{width}d}| {line.rstrip(chr(13))}"
            for i, line in enumerate(lines, start=1)
        ]
        return '\n'.join(numbered), lines, width

    def _clean_tag(self, t: str) -> str:
        return t.strip().strip("<>").strip()

    def _locate(self, lines, i, needle, radius=8):
        # 1) exact on the stated line
        if 0 <= i < len(lines) and needle in lines[i]:
            return i
        # 2) nearby lines (LLM miscounts by a few lines, esp. around page breaks)
        for d in range(1, radius + 1):
            for j in (i - d, i + d):
                if 0 <= j < len(lines) and needle in lines[j]:
                    return j
        # 3) globally unique occurrence anywhere
        hits = [j for j, ln in enumerate(lines) if needle in ln]
        return hits[0] if len(hits) == 1 else None  # None = ambiguous or absent

    def reconstruct(self, lines, spans):
        n = len(lines)
        opens = {i: [] for i in range(n)}
        closes = {i: [] for i in range(n)}
        sublines = []  # (order, stated_line_idx, tag, needle) — line may drift

        for order, s in enumerate(spans):
            a, b, tag = s['start_line'] - 1, s['end_line'] - 1, self._clean_tag(s['tag'])
            if s.get('text') is not None:
                if a != b:
                    logger.warning("Sub-line span %r has start_line != end_line (%d..%d); "
                                   "using start_line", s['text'], a + 1, b + 1)
                sublines.append((order, a, tag, s['text']))
            else:
                opens[a].append((order, b - a, tag))
                closes[b].append((order, b - a, tag))

        # Mutable per-line buffer that sub-line tags are written into.
        out_lines = list(lines)

        # Pass 1: place sub-line tags, tolerating line-number drift.
        for order, i, tag, needle in sorted(sublines, key=lambda x: x[0]):
            j = self._locate(lines, i, needle)
            if j is None:
                logger.warning("Skipping unplaceable sub-line tag %r (stated line %d)",
                               needle, i + 1)
                continue
            if j != i:
                logger.info("Relocated sub-line %r from line %d to %d", needle, i + 1, j + 1)
            body = out_lines[j]
            idx = body.find(needle)
            if idx == -1:
                # Present in the original line but a prior tag on the same line
                # overlapped/consumed it. Degrade gracefully.
                logger.warning("Sub-line tag %r no longer findable on line %d after "
                               "prior tagging; skipping", needle, j + 1)
                continue
            out_lines[j] = body[:idx] + f'<{tag}>' + needle + f'</{tag}>' + body[idx + len(needle):]

        # Pass 2: wrap each (possibly sub-line-tagged) line with line-level tags.
        out = []
        for i in range(n):
            open_str = ''.join(f'<{t}>' for _, _, t in sorted(opens[i], key=lambda x: (-x[1], x[0])))
            close_str = ''.join(f'</{t}>' for _, _, t in sorted(closes[i], key=lambda x: (x[1], -x[0])))
            out.append(open_str + out_lines[i] + close_str)
        return '\n'.join(out)


    def analyze_single_entry(
        self,
        text: str,
        system_prompt: str,
        user_prompt_template: str,
        expected_schema: Dict[str, Any],
        text_limit: int = 8000,
        preprocess: bool = False,
        postprocess: bool = False,
    ) -> Dict[str, Any]:

        if preprocess:
            numbered_text, lines, _ = self.preprocess(text)
            limited_text = numbered_text
            user_prompt = user_prompt_template.format(text=limited_text, numbered_text=limited_text)
        else:
            user_prompt = user_prompt_template.format(text=text)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = retry_call(self._chat_model.invoke, messages, max_retries=self._max_retries,
                                  retry_delay=self._retry_delay)
            result = self._parse_json(response.content)

            if postprocess and 'spans' in result:
                result['tagged_text'] = self.reconstruct(lines, result.get('spans', []))

            return self._validate_result(result, expected_schema)

        except Exception as e:
            logger.exception("LLM analysis failed")
            raise RuntimeError(
                f"LLM analysis failed ({self._provider}:{self.model_name}): {e}"
            ) from e

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response with multiple fallback strategies."""
        if not isinstance(text, str):
            raise ValueError(f"Expected string response, got {type(text)}")

        original_text = text
        text = text.strip()

        if not text:
            raise ValueError("Empty response from LLM")

        # 1. Direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError(f"Expected JSON object, got {type(parsed)}")
        except json.JSONDecodeError:
            pass

        # 2. Strip markdown code fences
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fence_match:
            fenced_text = fence_match.group(1).strip()

            try:
                parsed = json.loads(fenced_text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

            # 2b. Repair fenced JSON
            try:
                repaired = json_repair.loads(fenced_text)
                if isinstance(repaired, dict):
                    return repaired
            except Exception:
                pass

        # 3. Find first valid JSON object embedded in text
        decoder = json.JSONDecoder()

        for i, char in enumerate(text):
            if char != "{":
                continue

            try:
                parsed, _ = decoder.raw_decode(text[i:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        # 4. Extract balanced {...} block
        json_block = self._extract_balanced_json_object(text)
        if json_block:
            try:
                parsed = json.loads(json_block)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

            # 4b. Repair extracted JSON block
            try:
                repaired = json_repair.loads(json_block)
                if isinstance(repaired, dict):
                    return repaired
            except Exception:
                pass

        # 5. Last resort: repair entire response
        try:
            repaired = json_repair.loads(text)
            if isinstance(repaired, dict):
                return repaired
        except Exception:
            pass

        raise ValueError(
            "Could not parse JSON from response. First 1000 chars:\n"
            f"{original_text[:1000]}"
        )

    def _extract_balanced_json_object(self, text: str) -> Optional[str]:
        """
        Extract the first balanced {...} JSON-like object from text.
        Handles braces inside strings.
        """

        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            char = text[i]

            if escape:
                escape = False
                continue

            if char == "\\":
                escape = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

                if depth == 0:
                    return text[start : i + 1]

        return None

    # Schema validation helpers
    def _validate_result(self, result: Dict[str, Any], expected_schema: Dict[str, Any]) -> Dict[str, Any]:
        validated_result = {}

        for key, schema_info in expected_schema.items():
            if isinstance(schema_info, dict) and "default" in schema_info:
                default_value = schema_info["default"]
                expected_type = schema_info.get("type", type(default_value))

                clean_key = key.strip()
                value = None

                for result_key in result.keys():
                    if result_key.strip() == clean_key:
                        value = result[result_key]
                        break

                if value is not None:
                    if expected_type == list:
                        if isinstance(value, list):
                            validated_result[clean_key] = value
                        elif value:
                            validated_result[clean_key] = [value]
                        else:
                            validated_result[clean_key] = default_value
                    else:
                        validated_result[clean_key] = value if isinstance(value, expected_type) else default_value
                else:
                    validated_result[clean_key] = default_value
            else:
                clean_key = key.strip()
                value = None
                for result_key in result.keys():
                    if result_key.strip() == clean_key:
                        value = result[result_key]
                        break
                validated_result[clean_key] = value if value is not None else schema_info

        return validated_result
