"""
NER Model Management

This module handles loading and caching of NER models with lazy initialization.
"""

import logging
import spacy
from typing import Dict, Any
from transformers import pipeline
from .ner_config import NER_MODELS
from flair.models import SequenceTagger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton class to manage NER model loading and caching.
    
    This class ensures models are loaded only once and cached for reuse,
    improving performance and memory usage.
    """
    
    _instance = None
    _models: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_spacy_model(self, language: str):
        """
        Load and cache a spaCy model for the specified language.
        
        Args:
            language: Language code ('de', 'nl', 'en')
            
        Returns:
            Loaded spaCy model, or None if language not supported or model fails to load
        """
        if language not in NER_MODELS['spacy']:
            logger.warning(f"spaCy model not available for language '{language}'")
            return None
        
        model_name = NER_MODELS['spacy'][language]
        model_key = f"spacy_{language}"
        
        if model_key not in self._models:
            try:
                self._models[model_key] = spacy.load(model_name)
            except OSError as e:
                logger.warning(f"spaCy model '{model_name}' could not be loaded: {e}")
                return None
        
        return self._models[model_key]
    
    def get_flair_model(self, language: str = None, model_name: str = None):
        """
        Load and cache a Flair model.
        
        Args:
            language: Language code ('de', 'nl', 'en') - uses config if provided
            model_name: Flair model name (e.g., 'flair/ner-german-legal') - overrides language if provided
            
        Returns:
            Loaded Flair SequenceTagger model, or None if not available
        """
        # If model_name is provided, use it directly; otherwise use language from config
        if model_name:
            final_model_name = model_name
        elif language and language in NER_MODELS['flair']:
            final_model_name = NER_MODELS['flair'][language]
        else:
            logger.warning(f"Flair model not available for language '{language}'")
            return None
        
        model_key = f"flair_{final_model_name.replace('/', '_')}"
        
        if model_key not in self._models:
            try:
                self._models[model_key] = SequenceTagger.load(final_model_name)
            except Exception as e:
                logger.warning(f"Flair model '{final_model_name}' could not be loaded: {e}")
                return None
        
        return self._models[model_key]
    
    def get_huggingface_model(self, language: str):
        """
        Load and cache the Hugging Face NER model for the specified language.
        
        Args:
            language: Language code ('nl', 'en')
            
        Returns:
            Loaded Hugging Face pipeline for token classification, or None if language not supported
        """
        if language not in NER_MODELS['huggingface']:
            return None
        
        model_name = NER_MODELS['huggingface'][language]
        aggregation_strategy = NER_MODELS['huggingface'].get('aggregation_strategy', 'simple')
        
        # Cache by model name (same model may serve multiple languages)
        model_key = f"huggingface_{model_name.replace('/', '_')}"
        
        if model_key not in self._models:
            try:
                self._models[model_key] = pipeline(
                    "token-classification",
                    model=model_name,
                    aggregation_strategy=aggregation_strategy
                )
            except Exception as e:
                logger.warning(f"HuggingFace model '{model_name}' could not be loaded: {e}")
                return None
        
        return self._models[model_key]
    
    def get_title_extraction_model(self):
        """
        Load and cache the title extraction model (Gemma).
        
        Returns:
            Loaded Hugging Face pipeline for text generation
            
        Raises:
            Exception: If model cannot be loaded
        """
        model_key = "title_extraction_pipeline"
        
        if model_key not in self._models:
            try:                
                model_name = NER_MODELS['title_extraction']['model']
                
                # Explicitly load tokenizer and model first
                print(f"Loading title extraction model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    device_map="cpu"
                )
                
                # Create pipeline from the loaded model and tokenizer
                self._models[model_key] = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    #device="cpu"
                )
                print(f"Successfully loaded title extraction model")
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                raise Exception(
                    f"Title extraction model could not be loaded. "
                    f"Error: {str(e)}\n{error_details}"
                )
        
        return self._models[model_key]
    
    
    def get_refinement_model(self):
        """
        Load and cache the entity refinement model (Longformer classifier).
        
        This model refines generic entity labels (DATE, LOCATION) into more 
        specific types (publication_date, impact_location, etc.).
        
        Returns:
            Tuple of (model, tokenizer) for the refinement classifier
            
        Raises:
            Exception: If model cannot be loaded
        """
        model_key = "refinement_model"
        tokenizer_key = "refinement_tokenizer"
        
        if model_key not in self._models:
            try:
                model_name = NER_MODELS['refinement']['model']
                
                logger.info(f"Loading entity refinement model: {model_name}")
                
                # Load tokenizer with special entity markers
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Load the sequence classification model
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                self._models[model_key] = model
                self._models[tokenizer_key] = tokenizer
                
                logger.info(f"Successfully loaded entity refinement model")
            except Exception as e:
                logger.warning(f"Entity refinement model could not be loaded: {e}")
                return None, None
        
        return self._models[model_key], self._models[tokenizer_key]
    
    def clear_cache(self):
        """Clear all cached models to free memory."""
        self._models.clear()


# Global model manager instance
model_manager = ModelManager()
