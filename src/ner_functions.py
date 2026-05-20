"""
Simplified NER Functions Interface

This module provides a clean, simple interface to the refactored NER system.
It maintains backward compatibility while using the improved architecture.
"""

import logging
from typing import List, Dict, Any, Optional
from functools import cache

from .config import get_config
from .ner_extractors import (
    create_german_composite_extractor,
    create_dutch_composite_extractor,
    create_english_composite_extractor,
    SpacyExtractor,
    HuggingFaceExtractor,
    FlairExtractor,
    LanguageRegexExtractor,
    CompositeExtractor,
    EntityRefiner
)

logger = logging.getLogger(__name__)


def get_composite_extractor(language: str) -> CompositeExtractor:
    if language == 'de':
        return create_german_composite_extractor()
    elif language == 'nl':
        return create_dutch_composite_extractor()
    elif language == 'en':
        return create_english_composite_extractor()
    else:
        raise ValueError(
            f"Unsupported language '{language}' for composite extractor "
            f"(supported: 'de', 'nl', 'en')"
        )


@cache
def get_extractor(language: str, extractor_type: str = 'composite'):
    """
    Get a cached extractor for the specified language and type.
    
    Args:
        language: Language code ('de', 'nl', 'en')
        extractor_type: Type of extractor ('composite', 'spacy', 'huggingface', 'flair', 'regex')
        
    Returns:
        Configured extractor instance

    Raises:
        ValueError: If extractor_type is not one of the supported types.
    """
    extractors = {
        'spacy': SpacyExtractor,
        'huggingface': HuggingFaceExtractor,
        'flair': FlairExtractor,
        'regex': LanguageRegexExtractor,
        'composite': get_composite_extractor
    }

    extractor = extractors.get(extractor_type)
    if extractor is None:
        raise ValueError(
            f"Unsupported extractor type '{extractor_type}' for language '{language}' "
            f"(supported: {sorted(extractors.keys())})"
        )
    return extractor(language)


def extract_entities(text: str, language: str = None, method: str = None, refine: bool = None) -> List[Dict[str, Any]]:
    """
    Extract entities from text using the specified language and method.
    
    Args:
        text: Input text to process
        language: Language of the text ('de', 'nl', 'en'). Defaults to config.ner.language.
        method: Extraction method ('composite', 'spacy', 'huggingface', 'flair', 'regex', 'title'). 
                Defaults to config.ner.method.
        refine: Whether to apply entity refinement to classify generic labels (DATE, LOCATION)
                into specific types (publication_date, impact_location, etc.).
                Defaults to DEFAULT_SETTINGS['enable_refinement'].
        
    Returns:
        List of entity dictionaries with keys: text, label, start, end.

    Raises:
        ValueError: If method or language is not supported.
        Exception: Propagates any extractor or refiner failure to the caller
            so the task framework can mark the task as failed.
    """
    # Use defaults from config if not provided
    config = get_config()
    if language is None:
        language = config.ner.language
    if method is None:
        method = config.ner.method
    if refine is None:
        refine = config.ner.enable_refinement

    # Get extractor (cached). Raises ValueError on unsupported method/language.
    extractor = get_extractor(language, method)

    entities = extractor.extract(text)

    # Apply refinement if enabled
    if refine and entities:
        refiner = EntityRefiner()
        entities = refiner.refine(entities, text)
        logger.debug(f"Applied entity refinement to {len(entities)} entities")

    return entities
