"""
Simplified NER Functions Interface

This module provides a clean, simple interface to the refactored NER system.
It maintains backward compatibility while using the improved architecture.
"""

from typing import List, Dict, Any
from functools import cache

from .ner_extractors import (
    create_german_extractor,
    create_dutch_extractor,
    create_english_extractor,
    SpacyExtractor,
    FlairExtractor,
    LanguageRegexExtractor,
    TitleExtractor,
    CompositeExtractor
)


def get_composite_extractor(language: str) -> CompositeExtractor:
    if language == 'german':
        return create_german_extractor()
    elif language == 'dutch':
        return create_dutch_extractor()
    elif language == 'english':
        return create_english_extractor()
    else:
        raise ValueError(f"Unsupported language: {language}")


@cache
def get_extractor(language: str, extractor_type: str = 'composite'):
    """
    Get a cached extractor for the specified language and type.
    
    Args:
        language: Language code ('german', 'dutch', 'english')
        extractor_type: Type of extractor ('composite', 'spacy', 'flair', 'regex', 'title')
        
    Returns:
        Configured extractor instance
    """

    extractors = {
        'spacy': SpacyExtractor,
        'flair': FlairExtractor,
        'regex': LanguageRegexExtractor,
        'title': TitleExtractor,
        'composite': get_composite_extractor
    }

    extractor = extractors.get(extractor_type)
    if extractor is not None:
        return extractor(language)
    raise ValueError(f"Unsupported combination: {language} + {extractor_type}")

# New simplified interface
def extract_entities(text: str, language: str = 'german', method: str = 'composite') -> List[Dict[str, Any]]:
    """
    Extract entities from text using the specified language and method.
    
    Args:
        text: Input text to process
        language: Language of the text ('german', 'dutch', 'english')
        method: Extraction method ('composite', 'spacy', 'flair', 'regex', 'title')
        
    Returns:
        List of entity dictionaries with keys: text, label, start, end
        
    Example:
        entities = extract_entities("John Doe works at Microsoft in Berlin.", 'english')
        # For German legal text using Flair:
        entities = extract_entities("Herr W. verstieß gegen § 36 Abs. 7 IfSG.", 'german', 'flair')
        # For title extraction:
        entities = extract_entities(document_text, 'dutch', 'title')
    """
    if method == 'composite':
        extractor = get_extractor(language, 'composite')
    elif method == 'spacy':
        extractor = get_extractor(language, 'spacy')
    elif method == 'flair':
        extractor = get_extractor(language, 'flair')
    elif method == 'regex':
        extractor = get_extractor(language, 'regex')
    elif method == 'title':
        extractor = get_extractor(language, 'title')
    else:
        raise ValueError(f"Unsupported method '{method}' for language '{language}'")
    
    return extractor.extract(text)
