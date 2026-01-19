"""
Simplified NER Functions Interface

This module provides a clean, simple interface to the refactored NER system.
It maintains backward compatibility while using the improved architecture.
"""

import logging
from typing import List, Dict, Any, Optional
from functools import cache

from .ner_config import DEFAULT_SETTINGS
from .ner_extractors import (
    create_german_composite_extractor,
    create_dutch_composite_extractor,
    create_english_composite_extractor,
    SpacyExtractor,
    HuggingFaceExtractor,
    FlairExtractor,
    LanguageRegexExtractor,
    TitleExtractor,
    CompositeExtractor,
    EntityRefiner
)

logger = logging.getLogger(__name__)


def get_composite_extractor(language: str) -> Optional[CompositeExtractor]:
    if language == 'de':
        return create_german_composite_extractor()
    elif language == 'nl':
        return create_dutch_composite_extractor()
    elif language == 'en':
        return create_english_composite_extractor()
    else:
        logger.warning(f"Unsupported language '{language}' for composite extractor")
        return None


@cache
def get_extractor(language: str, extractor_type: str = 'composite'):
    """
    Get a cached extractor for the specified language and type.
    
    Args:
        language: Language code ('de', 'nl', 'en')
        extractor_type: Type of extractor ('composite', 'spacy', 'huggingface', 'flair', 'regex', 'title')
        
    Returns:
        Configured extractor instance
    """
    extractors = {
        'spacy': SpacyExtractor,
        'huggingface': HuggingFaceExtractor,
        'flair': FlairExtractor,
        'regex': LanguageRegexExtractor,
        'title': TitleExtractor,
        'composite': get_composite_extractor
    }

    extractor = extractors.get(extractor_type)
    if extractor is not None:
        return extractor(language)
    
    logger.warning(f"Unsupported extractor type '{extractor_type}', returning None")
    return None


def extract_entities(text: str, language: str = None, method: str = None, refine: bool = None) -> List[Dict[str, Any]]:
    """
    Extract entities from text using the specified language and method.
    
    Args:
        text: Input text to process
        language: Language of the text ('de', 'nl', 'en'). Defaults to DEFAULT_SETTINGS['language'].
        method: Extraction method ('composite', 'spacy', 'huggingface', 'flair', 'regex', 'title'). 
                Defaults to DEFAULT_SETTINGS['method'].
        refine: Whether to apply entity refinement to classify generic labels (DATE, LOCATION)
                into specific types (publication_date, impact_location, etc.).
                Defaults to DEFAULT_SETTINGS['enable_refinement'].
        
    Returns:
        List of entity dictionaries with keys: text, label, start, end
        Returns empty list if extraction fails or method/language is unsupported.
    """
    # Use defaults from config if not provided
    if language is None:
        language = DEFAULT_SETTINGS['language']
    if method is None:
        method = DEFAULT_SETTINGS['method']
    if refine is None:
        refine = DEFAULT_SETTINGS['enable_refinement']
    
    # Get extractor (cached)
    extractor = get_extractor(language, method)
    
    if extractor is None:
        logger.warning(f"Unsupported method '{method}' for language '{language}', returning empty result")
        return []
    
    try:
        entities = extractor.extract(text)
        
        # Apply refinement if enabled
        if refine and entities:
            refiner = EntityRefiner()
            entities = refiner.refine(entities, text)
            logger.debug(f"Applied entity refinement to {len(entities)} entities")
        
        return entities
    except Exception as e:
        logger.warning(f"Entity extraction failed ({method}/{language}): {e}")
        return []
