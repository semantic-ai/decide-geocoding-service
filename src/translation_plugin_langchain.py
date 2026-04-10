"""
LangChain Translation Plugin for translatepy.

Provider-agnostic LLM translation using LangChain's init_chat_model factory.
Supports any LangChain-compatible provider (ollama, openai, mistral, azure_openai, …)
by setting config.translation.langchain.provider in config.json.
"""

import logging
import re
from typing import Dict

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

from translatepy.language import Language
from translatepy.models import TranslationResult
from translatepy.translators.base import BaseTranslator

from .config import get_config


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
    "Keep dates and times in the exact same format as the source: "
    "if the source uses DD/MM/YYYY write DD/MM/YYYY, "
    "if the source uses DD maand YYYY write DD Month YYYY, "
    "never convert between numeric and written-out month names.\n"
    "Keep all other numbers exactly as written.\n"
    "Keep the legal meaning precise and formal.\n"
    "Preserve paragraph breaks and list structure where possible.\n"
    "Return only the translated text.\n"
    "Do NOT add notes, explanations, labels, quotation marks, or markdown fences."
)


class LangChainTranslateService(BaseTranslator):
    """Provider-agnostic LangChain-backed translator."""

    def __init__(self, **kwargs):
        if BaseTranslator != object:
            super().__init__(**kwargs)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = get_config().translation.langchain

        model_kwargs = {"temperature": self.config.temperature}
        if self.config.api_key:
            model_kwargs["api_key"] = self.config.api_key.get_secret_value()
        if self.config.base_url:
            model_kwargs["base_url"] = self.config.base_url
        if self.config.timeout:
            model_kwargs["timeout"] = self.config.timeout

        self._chat_model = init_chat_model(
            f"{self.config.provider}:{self.config.model_name}",
            **model_kwargs,
        )

        self.logger.info(
            "LangChain translation plugin initialised: provider=%s model=%s",
            self.config.provider,
            self.config.model_name,
        )

    # ------------------------------------------------------------------
    # Language helpers (identical to translation_plugin_ollama.py)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Text chunking (identical to translation_plugin_ollama.py)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

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
        response = self._chat_model.invoke(messages)
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
