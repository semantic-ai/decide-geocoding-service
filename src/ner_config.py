"""
NER Configuration and Constants

This module contains all configuration settings, model names, and constants
used by the NER extraction system.
"""

# Model Configuration
NER_MODELS = {
    'spacy': {
        'dutch': 'nl_core_news_sm',
        'german': 'de_core_news_sm',
        'english': 'en_core_web_sm'
    },
    'title_extraction': {
        'model': 'javdrher/decide-gemma3-270m',
        'max_new_tokens': 4000
    }
}

# Language-specific regex patterns organized by pattern type
# This structure allows easy extension: just add new pattern types like 'person', 'address', etc.
REGEX_PATTERNS = {
    'german': 
    {
        'date': [
        # 02.04.2025 or 2.4.2025
        r"\b([0-3]?\d)\.([0-1]?\d)\.(\d{4})\b",
        # 02-04-2025
        r"\b([0-3]?\d)-([0-1]?\d)-(\d{4})\b",
        # 2. April 2025 / 02. April 2025
        r"\b([0-3]?\d)\.\s*(Januar|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+(\d{4})\b",
        # April 2025 (month-year)
        r"\b(Januar|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+(\d{4})\b"
    ]},
    'dutch': {
        'date': [
        # 02.04.2025 or 2.4.2025
        r"\b([0-3]?\d)\.([0-1]?\d)\.(\d{4})\b",
        # 02-04-2025
        r"\b([0-3]?\d)-([0-1]?\d)-(\d{4})\b",
        # 02 April 2025 / 2 April 2025
        r"\b([0-3]?\d)\.\s*(Januari|Februari|Maart|April|Mei|Juni|Juli|Augustus|September|Oktober|November|December)\s+(\d{4})\b",
        # April 2025 (month-year)
        r"\b(Januari|Februari|Maart|April|Mei|Juni|Juli|Augustus|September|Oktober|November|December)\s+(\d{4})\b"
    ]}
}

# Title extraction instruction for Gemma model
# Works for both Dutch and German legal documents
TITLE_EXTRACTION_INSTRUCTION = """
    Your task is to generate responses in JSON format.
    Ensure that your output strictly follows the provided JSON structure.
    Each key in the JSON should be correctly populated according to the instructions given.
    Pay attention to details and ensure the JSON is well-formed and valid.
    Only use phrases present in the given text.
    Extract the title from the following text and return it in JSON format with the key \"title\".
    The title should be the main heading or subject of the document.
"""

# Default extraction settings
DEFAULT_SETTINGS = {
    'language': 'dutch',
    'method': 'regex',
    'deduplicate': True,
    'min_confidence': 0.5,
    'max_entities': 1000
}
