"""
HuggingFace Translation Plugin for translatepy
Uses Helsinki-NLP OPUS-MT models for high-quality translation.
"""

import re
import logging
from transformers import MarianMTModel, MarianTokenizer

from translatepy.translators.base import BaseTranslator
from translatepy.models import TranslationResult
from translatepy.language import Language


# Helsinki-NLP model mapping: (source_lang, target_lang) -> model_name
HELSINKI_MODELS = {
    ('nl', 'en'): 'Helsinki-NLP/opus-mt-nl-en',
    ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
}


class HuggingFaceTranslateService(BaseTranslator):
    """Helsinki-NLP OPUS-MT translator service."""
    
    def __init__(self, **kwargs):
        if BaseTranslator != object:
            super().__init__(**kwargs)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._models = {}  # Cache: (src, tgt) -> (model, tokenizer)
    
    def _get_lang_code(self, language):
        """Get ISO 639-1 language code."""
        if isinstance(language, Language):
            code = language.alpha2 if hasattr(language, 'alpha2') else str(language)
        else:
            code = str(language)
        
        # Normalize common variations
        code_map = {'nld': 'nl', 'eng': 'en', 'deu': 'de', 'fra': 'fr', 'spa': 'es', 'ita': 'it'}
        return code_map.get(code.lower(), code.lower()[:2])
    
    def _load_model(self, src_lang: str, tgt_lang: str):
        """Load Helsinki-NLP model for language pair."""
        cache_key = (src_lang, tgt_lang)
        
        if cache_key not in self._models:
            model_name = HELSINKI_MODELS.get(cache_key)
            
            if not model_name:
                available = ', '.join([f"{s}→{t}" for s, t in HELSINKI_MODELS.keys()])
                raise ValueError(f"No Helsinki-NLP model for {src_lang}→{tgt_lang}. Available: {available}")
            
            self.logger.info(f"Loading translation model: {model_name}")
            self._tokenizer = MarianTokenizer.from_pretrained(model_name)
            self._model = MarianMTModel.from_pretrained(model_name)
            self._models[cache_key] = (self._model, self._tokenizer)
            self.logger.info(f"Model loaded: {src_lang} → {tgt_lang}")
        
        return self._models[cache_key]
    
    def _language_normalize(self, language):
        """Normalize language code to translatepy Language object."""
        if isinstance(language, Language):
            return language
        return Language(str(language))
    
    def _language_denormalize(self, language):
        """Convert Language object back to string code."""
        if isinstance(language, Language):
            return language.alpha2 if hasattr(language, 'alpha2') else str(language)
        return str(language)
    
    def _split_text_at_word_boundaries(self, text: str, tokenizer: MarianTokenizer, max_tokens: int = 256) -> list:
        """
        Split text into chunks at word boundaries, respecting token limits.
        Ensures chunks don't exceed max_tokens and splits only at whitespace.
        First tries sentence boundaries, then falls back to word boundaries.
        Aligned with eTranslation chunking logic for consistency.
        """
        # Split on: sentence boundaries, bullets, newlines, and punctuation without space
        parts = re.split(r'(?<=[.!?])\s+|(?<=[.!?])(?=[A-ZÀ-ÖØ-Ý])|\n+|•\s*|;\s*', text)
        parts = [p.strip() for p in parts if p.strip()]
        chunks = []
        cur = ""
        
        for p in parts:
            if not cur:
                # Validate first chunk - don't accept oversized parts blindly
                if len(tokenizer.encode(p, add_special_tokens=False)) <= max_tokens:
                    cur = p
                else:
                    # First part is too large, split by words
                    words = p.split()
                    for w in words:
                        if not cur:
                            cur = w
                        else:
                            test_text = cur + " " + w
                            if len(tokenizer.encode(test_text, add_special_tokens=False)) <= max_tokens:
                                cur = test_text
                            else:
                                if cur:
                                    chunks.append(cur)
                                cur = w
            else:
                # Check if adding this part would exceed token limit
                test_text = cur + " " + p
                token_count = len(tokenizer.encode(test_text, add_special_tokens=False))
                
                if token_count <= max_tokens:
                    cur = test_text
                else:
                    # Save current chunk before processing next part
                    if cur:
                        chunks.append(cur)
                    cur = ""
                    
                    # Check if the part itself exceeds limit (handle immediately like eTranslation)
                    part_token_count = len(tokenizer.encode(p, add_special_tokens=False))
                    if part_token_count > max_tokens:
                        # Word-level fallback for oversized parts
                        words = p.split()
                        for w in words:
                            # Safety check: if a single word exceeds limit, include it anyway
                            word_token_count = len(tokenizer.encode(w, add_special_tokens=False))
                            if word_token_count > max_tokens:
                                # Save current chunk if exists
                                if cur:
                                    chunks.append(cur)
                                    cur = ""
                                # Add oversized word as its own chunk (will be truncated by tokenizer)
                                chunks.append(w)
                                continue
                            
                            if not cur:
                                cur = w
                            else:
                                test_text = cur + " " + w
                                token_count = len(tokenizer.encode(test_text, add_special_tokens=False))
                                if token_count <= max_tokens:
                                    cur = test_text
                                else:
                                    if cur:
                                        chunks.append(cur)
                                    cur = w
                    else:
                        # Part fits alone, save current chunk and start new one
                        if cur:
                            chunks.append(cur)
                        cur = p
        
        if cur and cur.strip():
            chunks.append(cur.strip())
        
        # Filter out any empty chunks that might have been created
        chunks = [c for c in chunks if c.strip()]
        
        return chunks if chunks else [text]
    
    def _translate_chunked(self, text: str, tokenizer: MarianTokenizer, model: MarianMTModel, 
                          src_lang: str, tgt_lang: str) -> str:
        """Translate long text by chunking and combining results."""
        chunks = self._split_text_at_word_boundaries(text, tokenizer, max_tokens=256)
        
        # Filter empty chunks to prevent model from generating garbage
        chunks = [c for c in chunks if c.strip()]
        
        if len(chunks) == 1:
            # Single chunk, translate directly
            encoded = tokenizer(chunks[0], return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = model.generate(**encoded, max_new_tokens=1024)
            return tokenizer.decode(translated[0], skip_special_tokens=True)
        
        self.logger.info(f"Chunking text into {len(chunks)} chunks for translation")
        
        translated_chunks = []
        for i, chunk in enumerate(chunks, 1):
            self.logger.debug(f"Translating chunk {i}/{len(chunks)} ({len(chunk)} chars)")
            encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = model.generate(**encoded, max_new_tokens=1024)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            translated_chunks.append(translated_text)
        
        # Combine chunks - preserve structure with newlines, normalize excessive whitespace
        combined = "\n".join(translated_chunks)
        combined = re.sub(r'\n{3,}', '\n\n', combined)
        combined = re.sub(r'[ \t]+', ' ', combined)
        combined = combined.strip()
        
        return combined
    
    def _translate(self, text: str, destination_language: str, source_language: str = "auto") -> str:
        src_lang = self._get_lang_code(source_language)
        tgt_lang = self._get_lang_code(destination_language)
        
        self.logger.info(f"Translating {src_lang} → {tgt_lang} ({len(text)} chars)")
        
        model, tokenizer = self._load_model(src_lang, tgt_lang)
        
        # Check token count to determine if chunking is needed
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        
        if token_count > 256:
            self.logger.info(f"Text exceeds token limit ({token_count} > 512), using chunking")
            translation = self._translate_chunked(text, tokenizer, model, src_lang, tgt_lang)
        else:
            # Single chunk translation
            encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = model.generate(**encoded, max_new_tokens=1024)
            translation = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        self.logger.info(f"Translation completed: {len(translation)} characters")
        return translation
    
    def translate(self, text: str, destination_language, source_language="auto"):
        dest_lang = self._get_lang_code(destination_language)
        src_lang = self._get_lang_code(source_language)
        
        translated_text = self._translate(text, dest_lang, src_lang)
        
        return TranslationResult(
            service="Helsinki-NLP",
            source=text,
            result=translated_text,
            source_language=source_language if isinstance(source_language, Language) else Language(src_lang),
            destination_language=destination_language if isinstance(destination_language, Language) else Language(dest_lang)
        )

