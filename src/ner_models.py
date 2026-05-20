"""
NER Model Management

This module handles loading and caching of NER models with lazy initialization.
"""

import logging
import spacy
import torch
from typing import Dict, Any
from transformers import pipeline
from .ner_config import NER_MODELS
from flair.models import SequenceTagger
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
            Loaded spaCy model.

        Raises:
            ValueError: If no spaCy model is configured for the given language.
            RuntimeError: If the configured spaCy model cannot be loaded
                (e.g. the model package is not installed in the container).
        """
        if language not in NER_MODELS['spacy']:
            raise ValueError(
                f"No spaCy model configured for language '{language}' "
                f"(configured: {sorted(NER_MODELS['spacy'].keys())})"
            )

        model_name = NER_MODELS['spacy'][language]
        model_key = f"spacy_{language}"
        
        if model_key not in self._models:
            try:
                self._models[model_key] = spacy.load(model_name)
            except OSError as e:
                raise RuntimeError(
                    f"Could not load spaCy model '{model_name}' for language '{language}': {e}"
                ) from e

        return self._models[model_key]
    
    def get_flair_model(self, language: str = None, model_name: str = None):
        """
        Load and cache a Flair model.
        
        Args:
            language: Language code ('de', 'nl', 'en') - uses config if provided
            model_name: Flair model name (e.g., 'flair/ner-german-legal') - overrides language if provided
            
        Returns:
            Loaded Flair SequenceTagger model.

        Raises:
            ValueError: If neither model_name nor a configured language is provided.
            RuntimeError: If the Flair model cannot be downloaded or loaded.
        """
        # If model_name is provided, use it directly; otherwise use language from config
        if model_name:
            final_model_name = model_name
        elif language and language in NER_MODELS['flair']:
            final_model_name = NER_MODELS['flair'][language]
        else:
            raise ValueError(
                f"No Flair model configured for language '{language}' "
                f"(configured: {sorted(NER_MODELS['flair'].keys())})"
            )

        model_key = f"flair_{final_model_name.replace('/', '_')}"

        if model_key not in self._models:
            try:
                self._models[model_key] = SequenceTagger.load(final_model_name)
            except Exception as e:
                raise RuntimeError(
                    f"Could not load Flair model '{final_model_name}': {e}"
                ) from e

        return self._models[model_key]
    
    def get_huggingface_model(self, language: str):
        """
        Load and cache the Hugging Face NER model for the specified language.
        
        Args:
            language: Language code ('nl', 'en')
            
        Returns:
            Loaded Hugging Face pipeline for token classification.

        Raises:
            ValueError: If no HuggingFace model is configured for the language.
            RuntimeError: If the configured model cannot be downloaded or loaded.
        """
        if language not in NER_MODELS['huggingface']:
            configured = [k for k in NER_MODELS['huggingface'].keys() if k != 'aggregation_strategy']
            raise ValueError(
                f"No HuggingFace model configured for language '{language}' "
                f"(configured: {sorted(configured)})"
            )

        model_name = NER_MODELS['huggingface'][language]
        aggregation_strategy = NER_MODELS['huggingface'].get('aggregation_strategy', 'simple')
        
        # Cache by model name (same model may serve multiple languages)
        model_key = f"huggingface_{model_name.replace('/', '_')}"
        
        if model_key not in self._models:
            try:
                self._models[model_key] = pipeline(
                    "token-classification",
                    model=model_name,
                    aggregation_strategy=aggregation_strategy,
                    device="cpu",
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not load HuggingFace model '{model_name}' for language '{language}': {e}"
                ) from e

        return self._models[model_key]
    
    def get_refinement_model(self):
        """
        Load and cache the entity refinement model (Longformer classifier).
        
        This model refines generic entity labels (DATE, LOCATION) into more 
        specific types (publication_date, impact_location, etc.).
        
        Returns:
            Tuple of (model, tokenizer) for the refinement classifier
            
        Raises:
            RuntimeError: If the refinement model cannot be downloaded or loaded.
        """
        model_key = "refinement_model"
        tokenizer_key = "refinement_tokenizer"
        
        if model_key not in self._models:
            model_name = NER_MODELS['refinement']['model']
            try:
                logger.info(f"Loading entity refinement model: {model_name}")
                
                # Load tokenizer with special entity markers
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Load the sequence classification model
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                self._models[model_key] = model
                self._models[tokenizer_key] = tokenizer
                
                logger.info(f"Successfully loaded entity refinement model")
            except Exception as e:
                raise RuntimeError(
                    f"Could not load entity refinement model '{model_name}': {e}"
                ) from e

        return self._models[model_key], self._models[tokenizer_key]
    
    def clear_cache(self):
        """Clear all cached models to free memory."""
        self._models.clear()


# Global model manager instance
model_manager = ModelManager()
