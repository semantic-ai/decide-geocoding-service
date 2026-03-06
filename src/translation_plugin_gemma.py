"""
Gemma Translation Plugin for translatepy
Uses javdrher/decide-gemma3-270m model for translation.
"""

import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from translatepy.translators.base import BaseTranslator
from translatepy.models import TranslationResult
from translatepy.language import Language

from .ner_config import NER_MODELS


class GemmaTranslateService(BaseTranslator):
    """Gemma model translator service."""
    
    def __init__(self, **kwargs):
        if BaseTranslator != object:
            super().__init__(**kwargs)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model = None
        self._tokenizer = None
        self._model_name = NER_MODELS['title_extraction']['model']
        self.logger.info("Gemma translation plugin initialized")
    
    def _load_model(self):
        """Load Gemma model and tokenizer."""
        if self._model is None or self._tokenizer is None:
            self.logger.info(f"Loading translation model: {self._model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                trust_remote_code=True,
                device_map=device
            )
            self.logger.info(f"Model loaded on {device}")
    
    def _get_lang_code(self, language):
        """Get ISO 639-1 language code."""
        if isinstance(language, Language):
            code = language.alpha2 if hasattr(language, 'alpha2') else str(language)
        else:
            code = str(language)
        
        code_map = {'nld': 'nl', 'eng': 'en', 'deu': 'de', 'fra': 'fr', 'spa': 'es', 'ita': 'it'}
        return code_map.get(code.lower(), code.lower()[:2])
    
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
        """Translate text using Gemma model."""
        self._load_model()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        src_lang = self._get_lang_code(source_language)
        tgt_lang = self._get_lang_code(destination_language)
        
        self.logger.info(f"Translating {src_lang} → {tgt_lang} ({len(text)} chars)")
        
        lang_names = {'nl': 'Dutch', 'en': 'English', 'de': 'German', 'fr': 'French', 'es': 'Spanish', 'it': 'Italian'}
        src_lang_name = lang_names.get(src_lang.lower(), src_lang.upper())
        tgt_lang_name = lang_names.get(tgt_lang.lower(), tgt_lang.upper())
        
        try:
            context = json.loads(text) if text.strip().startswith('{') else None
        except json.JSONDecodeError:
            context = None
        
        if not context or 'uri' not in context or 'title' not in context:
            context = {
                "uri": "Do not translate",
                "title": "",
                "description": text
            }
        
        context_str = json.dumps(context, ensure_ascii=False, indent=2)
        
        prompt = f"""<start_of_turn>user
Your task is to generate responses in JSON format. Ensure that your output strictly follows the provided JSON structure. Each key in the JSON should be correctly populated according to the instructions given. Pay attention to details and ensure the JSON is well-formed and valid.

####
Context: {context_str}
####
Task: Translate all fields of the item in the context to {tgt_lang_name}, except for the 'uri' field and any field explicitly marked as 'do not translate'. Place names and IDs should remain in their original language. Return the translated text as a JSON object with the following format:

    {{
    "uri": "Do not translate",
    "title": "Translated title",
    "description": "Translated description",
    "source": "{src_lang_name}",
    "target": "{tgt_lang_name}"}}<end_of_turn>
<start_of_turn>model"""
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(device)
        input_token_count = inputs['input_ids'].shape[1]
        max_new_tokens = int(input_token_count * 1.5)
        
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            top_k=64,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id if self._tokenizer.eos_token_id else self._tokenizer.pad_token_id,
            num_beams=1,
        )
        
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        if "<start_of_turn>model" in generated_text:
            response = generated_text[generated_text.find("<start_of_turn>model") + len("<start_of_turn>model"):].strip()
        else:
            response = generated_text
        
        if '{' in response and '}' in response:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            
            try:
                result = json.loads(json_str)
                translated_parts = []
                if result.get('title') and result.get('title') not in ["Translated title", "Do not translate", ""]:
                    translated_parts.append(result['title'])
                if result.get('description') and result.get('description') not in ["Translated description", "Do not translate", ""]:
                    translated_parts.append(result['description'])
                
                if translated_parts:
                    return "\n".join(translated_parts)
                elif result.get('text') and result.get('text') not in ["Translated text", "Do not translate", ""]:
                    return result['text']
            except json.JSONDecodeError:
                pass
        
        self.logger.warning("Could not extract translation from model response")
        return "Translation failed: Could not parse model response"
    
    def translate(self, text: str, destination_language, source_language="auto"):
        """Translate text and return TranslationResult."""
        dest_lang = self._get_lang_code(destination_language)
        src_lang = self._get_lang_code(source_language)
        
        translated_text = self._translate(text, dest_lang, src_lang)
        
        return TranslationResult(
            service="Gemma",
            source=text,
            result=translated_text,
            source_language=source_language if isinstance(source_language, Language) else Language(src_lang),
            destination_language=destination_language if isinstance(destination_language, Language) else Language(dest_lang)
        )

