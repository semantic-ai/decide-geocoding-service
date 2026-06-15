"""
LangChain Translation Plugin for translatepy.

Provider-agnostic LLM translation using LangChain's init_chat_model factory.
Supports any LangChain-compatible provider (ollama, openai, mistral, …)
by setting config.translation.langchain.provider in config.json.
"""

import re
from typing import Dict

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

from translatepy.language import Language
from translatepy.models import TranslationResult
from translatepy.translators.base import BaseTranslator
from helpers import logger

from .config import get_config
from .retry import retry_call


LANGUAGE_NAMES: Dict[str, str] = {
    "nl": "Dutch",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
}

SYSTEM_PROMPT = (
    "You are a legal-domain translator for municipal government decisions.\n"
    "Preserve official names, street names, person names, legal citations, article numbers, "
    "and identifiers exactly as written. Do not transliterate or literally translate proper nouns.\n"
    "Preserve dates in the exact format used in the source. "
    "If the source writes '2 februari 2026', write '2 February 2026' — "
    "translate only the month name to English, keep the day-month-year order with no leading zeros and no slashes. "
    "If the source writes '02/02/2026', keep '02/02/2026'. "
    "Never convert a written-out month date to a numeric date "
    "(e.g. never turn '2 februari 2026' into '02/02/2026').\n"
    "Keep all other numbers exactly as written.\n"
    "Keep the legal meaning precise and formal.\n"
    "Preserve paragraph breaks and list structure where possible.\n"
    "Return only the translated text.\n"
    "Do NOT add notes, explanations, labels, quotation marks, "
    "or any markdown formatting (no **bold**, no *italic*, no `backticks`, no code fences)."
)


class LangChainTranslateService(BaseTranslator):
    """Provider-agnostic LangChain-backed translator."""

    def __init__(self, **kwargs):
        if BaseTranslator != object:
            super().__init__(**kwargs)

        self.logger = logger
        self.config = get_config().translation.langchain

        model_kwargs = {"temperature": self.config.temperature}
        if self.config.api_key:
            model_kwargs["api_key"] = self.config.api_key.get_secret_value()
        if self.config.base_url:
            model_kwargs["base_url"] = self.config.base_url

        self._chat_model = init_chat_model(
            f"{self.config.provider}:{self.config.model_name}",
            **model_kwargs,
            max_retries=0,
        )

        self.logger.info(
            "LangChain translation plugin initialised: provider=%s model=%s",
            self.config.provider,
            self.config.model_name,
        )

    def _get_lang_code(self, language) -> str:
        if isinstance(language, Language):
            code = language.alpha2 if hasattr(language, "alpha2") else str(language)
        else:
            code = str(language)

        code_map = {"nld": "nl", "eng": "en", "deu": "de", "fra": "fr", "spa": "es"}
        return code_map.get(code.lower(), code.lower()[:2])

    def _language_normalize(self, language):
        if isinstance(language, Language):
            return language
        return Language(str(language))

    def _language_denormalize(self, language):
        if isinstance(language, Language):
            return language.alpha2 if hasattr(language, "alpha2") else str(language)
        return str(language)

    def _split_text_safely(self, text: str, max_chars: int) -> list[str]:
        """Split text into chunks that respect sentence boundaries."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        cur = ""

        for p in parts:
            if not cur:
                cur = p
            elif len(cur) + 1 + len(p) <= max_chars:
                cur += " " + p
            else:
                if len(p) > max_chars:
                    for w in p.split():
                        if not cur:
                            cur = w
                        elif len(cur) + 1 + len(w) <= max_chars:
                            cur += " " + w
                        else:
                            chunks.append(cur)
                            cur = w
                else:
                    chunks.append(cur)
                    cur = p

        if cur:
            chunks.append(cur)

        return chunks

    def _build_user_message(self, text: str, source_language: str, destination_language: str) -> str:
        src_name = LANGUAGE_NAMES.get(source_language.lower(), source_language.upper())
        dst_name = LANGUAGE_NAMES.get(destination_language.lower(), destination_language.upper())
        return (
            f"Translate the following text from {src_name} to {dst_name}.\n"
            f"If the input is already in {dst_name}, return it unchanged.\n\n"
            f"Input ({src_name}):\n{text}"
        )

    def _translate_chunk(self, text: str, destination_language: str, source_language: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=self._build_user_message(text, source_language, destination_language)),
        ]
        response = retry_call(self._chat_model.invoke, messages,max_retries=self.config.max_retries, retry_delay=self.config.retry_delay)
        translated = response.content.strip()
        if not translated:
            raise RuntimeError("LangChain translation returned an empty response")
        return translated

    def translate(self, text: str, destination_language, source_language="auto"):
        """Translate text and return a TranslationResult."""
        dest_lang = self._get_lang_code(destination_language)
        src_lang = self._get_lang_code(source_language)

        self.logger.info("Translating %s → %s (%s chars)", src_lang, dest_lang, len(text))

        chunks = self._split_text_safely(text, self.config.max_text_length)

        if len(chunks) == 1:
            translated_text = self._translate_chunk(chunks[0], dest_lang, src_lang)
        else:
            self.logger.info("Chunking text into %s chunks for translation", len(chunks))
            translated_chunks = []
            for index, chunk in enumerate(chunks, start=1):
                self.logger.debug(
                    "Translating chunk %s/%s (%s chars)", index, len(chunks), len(chunk)
                )
                translated_chunks.append(self._translate_chunk(chunk, dest_lang, src_lang))
            translated_text = " ".join(part.strip() for part in translated_chunks if part.strip())

        return TranslationResult(
            service=f"LangChain({self.config.provider}:{self.config.model_name})",
            source=text,
            result=translated_text,
            source_language=source_language if isinstance(source_language, Language) else Language(src_lang),
            destination_language=destination_language if isinstance(destination_language, Language) else Language(dest_lang),
        )
