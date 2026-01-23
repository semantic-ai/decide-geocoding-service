"""European Commission eTranslation plugin for translatepy."""

import re
import time
import logging
import requests
import threading
from typing import Dict, Tuple

from requests.auth import HTTPBasicAuth
from translatepy.language import Language
from translatepy.translators.base import BaseTranslator
from translatepy.models import TranslationResult

from .config import get_config


# Global callback storage and server for eTranslation callbacks
# Key: (requestId, targetLanguage) -> callback payload
_callback_storage: Dict[Tuple[int, str], Dict] = {}
_callback_lock = threading.Lock()


class ETRanslationService(BaseTranslator):
    """eTranslation adapter for translatepy."""

    def __init__(self, **kwargs):
        if BaseTranslator != object:
            super().__init__(**kwargs)

        self.logger = logging.getLogger(self.__class__.__name__)
        app_config = get_config()
        self.config = app_config.translation.etranslation
        
        # Validate authentication
        if not self.config.bearer_token and not (self.config.username and self.config.password):
            self.logger.warning("eTranslation credentials not configured in config.json (translation.etranslation)")
        
        # Setup callback base URL (must be publicly reachable by eTranslation)
        self.callback_base_url = self._setup_callback_url()
        
        # Guard: fail fast if callback URL is localhost
        if "localhost" in self.callback_base_url or "127.0.0.1" in self.callback_base_url:
            raise RuntimeError(
                f"Callback URL is localhost: {self.callback_base_url}. "
                f"eTranslation cannot reach localhost. Set translation.etranslation.callback_url to a public URL in config.json."
            )

    def _setup_callback_url(self) -> str:
        """Setup callback base URL from explicit config.

        This must resolve to a publicly reachable URL so that
        eTranslation can POST callbacks to it, as described in
        the official REST v2 documentation:
        https://language-tools.ec.europa.eu/dev-corner/etranslation/rest-v2/text
        """
        callback_url = self.config.callback_url
        if not callback_url:
            raise RuntimeError(
                "No public callback URL configured. "
                "Set translation.etranslation.callback_url in config.json "
                "(e.g., https://your-host/etranslation)"
            )
        
        # Normalize the callback URL
        callback_url = callback_url.rstrip('/')
        if callback_url.endswith('/callback'):
            callback_url = callback_url[:-9]
        
        self.logger.info(f"Using callback URL: {callback_url}")
        return callback_url

    def _get_lang_code(self, language) -> str:
        """Convert language to ISO 639-1 code string."""
        if isinstance(language, Language):
            return language.alpha2 if hasattr(language, "alpha2") else str(language)
        return str(language)

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
    
    def _split_text_safely(self, text: str, max_chars: int) -> list:
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
                    # Word-level fallback for oversized parts
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

    def _post_json(self, url: str, payload: dict, headers: dict, auth) -> requests.Response:
        """
        POST JSON with retry logic for transient errors (5xx, 429).
        Centralized retry helper.
        """
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, auth=auth, timeout=self.config.timeout_seconds)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as e:
                if e.response.status_code >= 500 or e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        self.logger.warning(f"API {e.response.status_code}, retrying in {wait_time:.1f}s ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                raise
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    self.logger.warning(f"Request failed: {e}, retrying in {wait_time:.1f}s ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                raise

    def _extract_translated_text(self, callback_data: dict) -> str:
        """
        Extract translated text from callback payload.
        Handles success/delivery/failure formats.
        Raises RuntimeError on failure.
        """
        # Check for failure first
        if callback_data.get("errorCode") or callback_data.get("errorMessage"):
            error_code = callback_data.get("errorCode")
            error_msg = callback_data.get("errorMessage", "Unknown error")
            raise RuntimeError(f"eTranslation callback error {error_code}: {error_msg}")
        
        # Extract translated text from various formats
        result_obj = callback_data.get("result")
        return (
            callback_data.get("translatedText") or 
            (result_obj if isinstance(result_obj, str) else None) or
            (isinstance(result_obj, dict) and result_obj.get("translatedText")) or
            ""
        )

    def _submit_translation_request(self, text: str, dest_lang: str, src_lang: str) -> int:
        """Submit translation request and return requestId."""
        url = f"{self.config.base_url}/askTranslate"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "translatepy-etranslation/0.1"
        }
        
        auth = None
        if self.config.bearer_token:
            headers["Authorization"] = f"Bearer {self.config.bearer_token.get_secret_value()}"
        else:
            password = self.config.password.get_secret_value() if self.config.password else None
            auth = HTTPBasicAuth(self.config.username, password)

        callback_url = f"{self.callback_base_url}/callback"
        payload = {
            "textToTranslate": text,
            "targetLanguages": [dest_lang.upper()],
            "sourceLanguage": src_lang.upper(),
            "domain": self.config.domain.upper(),
            "notifications": {
                "success": {"http": callback_url},
                "failure": {"http": callback_url}
            },
            "deliveries": {"http": callback_url},
            "callerInformation": {
                "externalReference": f"tp-{int(time.time() * 1000)}"
            }
        }

        response = self._post_json(url, payload, headers, auth)
        data = response.json()
        request_id = data.get("requestId")
        
        if not request_id:
            raise RuntimeError(f"eTranslation API did not return requestId. Response: {data}")
        if isinstance(request_id, int) and request_id < 0:
            raise RuntimeError(f"eTranslation API returned error code: {request_id}")
        
        return request_id

    def _wait_for_callback(self, request_id: int, target_lang: str) -> str:
        """Wait for translation callback and return translated text."""
        key = (int(request_id), target_lang.upper())
        start_time = time.time()
        poll_interval = 0.5
        
        while time.time() - start_time < self.config.callback_wait_timeout:
            with _callback_lock:
                if key in _callback_storage:
                    callback_data = _callback_storage.pop(key)
                    translated_text = self._extract_translated_text(callback_data)
                    if translated_text:
                        return translated_text
            
            time.sleep(poll_interval)
        
        raise RuntimeError(
            f"eTranslation callback timeout after {self.config.callback_wait_timeout}s. "
            f"Request ID: {request_id}, target: {target_lang}"
        )

    def _translate_chunked(self, text: str, dest_lang: str, src_lang: str, source_language, destination_language):
        """Translate long text by chunking, processing chunks in parallel."""
        pieces = self._split_text_safely(text, self.config.max_text_length)
        self.logger.info(f"Chunking text: {len(pieces)} chunks")
        
        # Submit all chunks first
        request_ids = []
        for i, chunk in enumerate(pieces, 1):
            request_id = self._submit_translation_request(chunk, dest_lang, src_lang)
            request_ids.append((i, request_id))
        
        # Wait for all callbacks in parallel
        translated_chunks = [None] * len(pieces)
        start_time = time.time()
        poll_interval = 0.5
        
        while time.time() - start_time < self.config.callback_wait_timeout:
            completed = 0
            with _callback_lock:
                for idx, request_id in request_ids:
                    if translated_chunks[idx - 1] is not None:
                        completed += 1
                        continue
                    
                    key = (int(request_id), dest_lang.upper())
                    if key in _callback_storage:
                        callback_data = _callback_storage.pop(key)
                        translated_text = self._extract_translated_text(callback_data)
                        if translated_text:
                            translated_chunks[idx - 1] = translated_text
                            completed += 1
            
            if completed == len(pieces):
                break
            time.sleep(poll_interval)
        
        if None in translated_chunks:
            missing = [i+1 for i, t in enumerate(translated_chunks) if t is None]
            raise RuntimeError(
                f"eTranslation callback timeout after {self.config.callback_wait_timeout}s. "
                f"Missing callbacks for chunks: {missing}"
            )
        
        stitched = re.sub(r'\s+', ' ', " ".join(translated_chunks)).strip()
        return TranslationResult(
            service="eTranslation",
            source=text,
            result=stitched,
            source_language=source_language if isinstance(source_language, Language) else Language(src_lang),
            destination_language=destination_language if isinstance(destination_language, Language) else Language(dest_lang),
        )

    def translate(self, text: str, destination_language, source_language: str = "auto"):
        """Translate text using eTranslation REST v2 API."""
        if not self.config.bearer_token and not (self.config.username and self.config.password):
            raise ValueError(
                "eTranslation credentials not configured. Set either bearer_token or username/password in config.json (translation.etranslation)"
            )
        
        dest_lang = self._get_lang_code(destination_language)
        src_lang = self._get_lang_code(source_language)

        if src_lang.lower() == "auto":
            raise ValueError(
                "eTranslation requires explicit sourceLanguage. "
                "Provide language code (e.g., 'EN', 'FR', 'NL')"
            )

        if len(text) > self.config.max_text_length:
            return self._translate_chunked(text, dest_lang, src_lang, source_language, destination_language)
        
        self.logger.info(f"Translation request: {src_lang} → {dest_lang} ({len(text)} chars)")
        
        request_id = self._submit_translation_request(text, dest_lang, src_lang)
        self.logger.info(f"Request ID: {request_id}")
        
        translated_text = self._wait_for_callback(request_id, dest_lang)
        
        return TranslationResult(
            service="eTranslation",
            source=text,
            result=translated_text,
            source_language=source_language if isinstance(source_language, Language) else Language(src_lang),
            destination_language=destination_language if isinstance(destination_language, Language) else Language(dest_lang),
        )
