"""
HuggingFace Translation Plugin for translatepy
Uses Helsinki-NLP OPUS-MT models for high-quality translation.
"""

import logging
from transformers import MarianMTModel, MarianTokenizer

from translatepy.translators.base import BaseTranslator
from translatepy.models import TranslationResult
from translatepy.language import Language


# Helsinki-NLP model mapping: (source_lang, target_lang) -> model_name
HELSINKI_MODELS = {
    ('nl', 'en'): 'Helsinki-NLP/opus-mt-nl-en',
    ('nl', 'de'): 'Helsinki-NLP/opus-mt-nl-de',
    ('nl', 'fr'): 'Helsinki-NLP/opus-mt-nl-fr',
    ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
    ('de', 'nl'): 'Helsinki-NLP/opus-mt-de-nl',
    ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en',
    ('fr', 'nl'): 'Helsinki-NLP/opus-mt-fr-nl',
    ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
    ('it', 'en'): 'Helsinki-NLP/opus-mt-it-en',
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
    
    def _translate(self, text: str, destination_language: str, source_language: str = "auto") -> str:
        src_lang = self._get_lang_code(source_language)
        tgt_lang = self._get_lang_code(destination_language)
        
        self.logger.info(f"Translating {src_lang} → {tgt_lang} ({len(text)} chars)")
        
        model, tokenizer = self._load_model(src_lang, tgt_lang)
        
        # Tokenize with truncation (MarianMT max ~512 tokens)
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate translation
        translated = model.generate(**encoded, max_length=512)
        
        # Decode
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

