import json
import logging
import re
from typing import Dict, Any, Optional
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
import json_repair

logger = logging.getLogger(__name__)


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
    ):
        self.model_name = model_name
        self._provider = provider

        kwargs: Dict[str, Any] = {"temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if json_mode:
            if provider == "ollama":
                kwargs["format"] = "json"
            #elif provider in {"mistralai", "mistral"}:
            #    kwargs["response_format"] = {"type": "json_object"}

        self._chat_model = init_chat_model(
            f"{provider}:{model_name}",
            **kwargs,
            timeout=600,
            max_retries=3
        )

    def analyze_single_entry_simple(
        self,
        text: str,
        system_prompt: str,
        user_prompt_template: str,
        output_model: BaseModel,
        text_limit: int = 8000,
    ) -> Dict[str, Any]:
        
        llm = self._chat_model.with_structured_output(
            output_model,
            method="function_calling",
            include_raw=True,
        )

        user_prompt = user_prompt_template.format(text=text[:text_limit])

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)

        if response["parsing_error"]:
            raise ValueError(
                f"Structured output parsing failed: {response['parsing_error']}"
            )

        parsed = response["parsed"]

        return parsed.model_dump()

    def analyze_single_entry(
        self,
        text: str,
        system_prompt: str,
        user_prompt_template: str,
        expected_schema: Dict[str, Any],
        text_limit: int = 8000,
    ) -> Dict[str, Any]:

        user_prompt = user_prompt_template.format(text=text[:text_limit])
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self._chat_model.invoke(messages)
            result = self._parse_json(response.content)
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
                repaired = json_repair.loads(bad_json)
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
                repaired = repair_json(json_block, return_objects=True)
                if isinstance(repaired, dict):
                    return repaired
            except Exception:
                pass

        # 5. Last resort: repair entire response
        try:
            repaired = repair_json(text, return_objects=True)
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
