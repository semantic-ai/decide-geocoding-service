import json
import re
from typing import Dict, Any, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage


class LLMAnalyzer:
    """
    Analyzer that routes LLM calls through LangChain's init_chat_model factory.
    Supports any provider recognised by LangChain (openai, ollama, mistral,
    azure_openai, …) purely through configuration — no if/else branching.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "mistral-nemo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self._provider = provider

        kwargs: Dict[str, Any] = {"temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._chat_model = init_chat_model(
            f"{provider}:{model_name}",
            **kwargs,
        )

    async def analyze_single_entry(
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
            response = await self._chat_model.ainvoke(messages)
            result = self._parse_json(response.content)
            return self._validate_result(result, expected_schema)

        except Exception as e:
            print(f"Analysis error: {e}")
            return self._create_error_result(expected_schema, f"Analysis failed: {str(e)}")

    # JSON parsing — three fallback strategies
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response with three fallback strategies."""
        text = text.strip()

        # 1. Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Strip markdown code fences
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. Regex: find first {...} block
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from response: {text[:200]}")

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

    def _create_error_result(self, expected_schema: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        error_result = {}

        for key, schema_info in expected_schema.items():
            clean_key = key.strip()
            if isinstance(schema_info, dict) and "default" in schema_info:
                error_result[clean_key] = schema_info["default"]
            else:
                error_result[clean_key] = schema_info

        error_result["error"] = error_message
        return error_result
