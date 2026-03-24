"""
Ollama Translation Plugin for translatepy
Uses an Ollama-hosted LLM for legal-domain translation.
"""

import logging
import re
import time

import requests

from translatepy.language import Language
from translatepy.models import TranslationResult
from translatepy.translators.base import BaseTranslator

from .config import get_config


LANGUAGE_NAMES = {
    "nl": "Dutch",
    "en": "English",
    "de": "German",
}


class OllamaTranslateService(BaseTranslator):
    """Ollama-backed translator service."""

    def __init__(self, **kwargs):
        if BaseTranslator != object:
            super().__init__(**kwargs)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = get_config().translation.ollama
        self.logger.info(
            "Ollama translation plugin initialized with model %s at %s",
            self.config.model,
            self.config.base_url,
        )

    def _get_lang_code(self, language):
        """Get ISO 639-1 language code."""
        if isinstance(language, Language):
            code = language.alpha2 if hasattr(language, "alpha2") else str(language)
        else:
            code = str(language)

        code_map = {
            "nld": "nl",
            "eng": "en",
            "deu": "de",
        }
        return code_map.get(code.lower(), code.lower()[:2])

    def _language_normalize(self, language):
        """Normalize language code to translatepy Language object."""
        if isinstance(language, Language):
            return language
        return Language(str(language))

    def _language_denormalize(self, language):
        """Convert Language object back to string code."""
        if isinstance(language, Language):
            return language.alpha2 if hasattr(language, "alpha2") else str(language)
        return str(language)

    def _split_text_safely(self, text: str, max_chars: int) -> list[str]:
        """Split text safely into chunks, preserving sentence boundaries."""
        parts = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        cur = ""

        for p in parts:
            if not cur:
                cur = p
            elif len(cur) + 1 + len(p) <= max_chars:
                cur += " " + p
            else:
                if len(p) > max_chars:
                    words = p.split()
                    for w in words:
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

    def _build_prompt(self, text: str, source_language: str, destination_language: str) -> str:
        """Build the legal-domain translation prompt for Ollama."""
        src_name = LANGUAGE_NAMES.get(source_language.lower(), source_language.upper())
        dst_name = LANGUAGE_NAMES.get(destination_language.lower(), destination_language.upper())

        return (
            "You are a legal-domain translator for municipal government decisions.\n"
            f"Translate the text from {src_name} to {dst_name}.\n"
            "Preserve official names, street names, person names, legal citations, article numbers, "
            "and identifiers exactly as written. Do not transliterate or literally translate proper nouns.\n"
            "Keep dates, times, numbers, and other non-textual information exactly as written.\n"
            "Keep the legal meaning precise and formal.\n"
            "Preserve paragraph breaks and list structure where possible.\n"
            f"If the input is already in {dst_name}, return it unchanged.\n"
            "Return only the translated text.\n"
            "Do NOT add notes, explanations, labels, quotation marks, or markdown fences.\n\n"
            f"Input ({src_name}):\n{text}"
        )

    def _post_json(self, url: str, payload: dict) -> requests.Response:
        """POST JSON with retry logic for transient Ollama failures."""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as exc:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    self.logger.warning(
                        "Ollama request failed: %s, retrying in %.1fs (%s/%s)",
                        exc,
                        wait_time,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait_time)
                    continue
                raise

    def _translate_chunk(self, text: str, destination_language: str, source_language: str) -> str:
        """Translate a single chunk via Ollama's generate endpoint."""
        prompt = self._build_prompt(text, source_language, destination_language)
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
            },
        }

        response = self._post_json(f"{self.config.base_url}/api/generate", payload)
        data = response.json()
        translated = data.get("response", "").strip()
        if not translated:
            raise RuntimeError(f"Ollama returned an empty translation response: {data}")
        return translated

    def translate(self, text: str, destination_language, source_language="auto"):
        """Translate text and return TranslationResult."""
        dest_lang = self._get_lang_code(destination_language)
        src_lang = self._get_lang_code(source_language)

        self.logger.info("Translating %s → %s (%s chars)", src_lang, dest_lang, len(text))

        chunks = self._split_text_safely(text, self.config.max_text_length)
        if len(chunks) == 1:
            translated_text = self._translate_chunk(chunks[0], dest_lang, src_lang)
        else:
            self.logger.info("Chunking text into %s chunks for Ollama translation", len(chunks))
            translated_chunks = []
            for index, chunk in enumerate(chunks, start=1):
                self.logger.debug(
                    "Translating chunk %s/%s (%s chars)",
                    index,
                    len(chunks),
                    len(chunk),
                )
                translated_chunks.append(self._translate_chunk(chunk, dest_lang, src_lang))

            translated_text = "\n\n".join(part.strip() for part in translated_chunks if part.strip())
            translated_text = re.sub(r"\n{3,}", "\n\n", translated_text).strip()

        return TranslationResult(
            service="Ollama",
            source=text,
            result=translated_text,
            source_language=source_language if isinstance(source_language, Language) else Language(src_lang),
            destination_language=destination_language if isinstance(destination_language, Language) else Language(dest_lang),
        )
