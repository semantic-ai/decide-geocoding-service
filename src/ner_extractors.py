"""
NER Entity Extractors

This module contains different NER extraction methods organized by approach.

Classes:
- SpacyGeoAnalyzer: Belgian location extraction (returns spaCy Doc)
- BaseExtractor: Base class for factory pattern extractors (returns dicts)
- SpacyExtractor, FlairExtractor, etc.: Factory pattern implementations
"""
import os
import re
import json
import logging
import spacy
from flair.data import Sentence
from typing import List, Dict, Any
from .ner_models import model_manager
from .ner_config import REGEX_PATTERNS, TITLE_EXTRACTION_INSTRUCTION, NER_MODELS, LABEL_MAPPINGS
from .config import get_config


# ============================================================================
# SPECIALIZED ANALYZER (Returns spaCy Doc for geocoding workflow)
# ============================================================================

class SpacyGeoAnalyzer:
    """
    Specialized analyzer for Belgian location extraction.
    
    NOTE: This is NOT part of the factory pattern (BaseExtractor).
    It returns raw spaCy Doc objects for compatibility with the geocoding
    workflow (process_text, form_addresses, form_locations).
    
    Use this for: Belgian location extraction with geocoding
    Use SpacyExtractor for: General-purpose NER with dict outputs
    """
    
    def __init__(self, model_path, labels=None):
        self.model_path = model_path
        self.labels = set(labels) if labels else None
        self.nlp = None
        self.logger = logging.getLogger(__name__)
        self.load_model()

    def load_model(self):
        """Load the spaCy NER model from the specified path."""
        if not os.path.exists(self.model_path):
            self.logger.error(f"Model not found at {self.model_path}")
            return
        try:
            self.nlp = spacy.load(self.model_path)
            self.logger.info(
                f"Loaded model: {self.nlp.meta.get('name', 'Unknown')} (v{self.nlp.meta.get('version', 'Unknown')})")
            if self.nlp.has_pipe("ner"):
                model_labels = self.nlp.get_pipe("ner").labels
                self.logger.info(f"Model labels: {model_labels}")
                if self.labels:
                    self.logger.info(
                        f"Filtering for labels: {sorted(self.labels)}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.nlp = None

    def extract_entities(self, text):
        """
        Extract named entities from text and return spaCy Doc object.
        
        Returns spaCy Doc for compatibility with form_addresses() and form_locations()
        in helper_functions.py which expect entity.label_ and entity.text attributes.
        """
        if not self.nlp:
            return {"error": "Model not loaded"}
        if not text.strip():
            return {"entities": [], "text": text}
        try:
            return self.nlp(text)
        except Exception as e:
            return {"error": f"Processing error: {e}", "text": text}


# ============================================================================
# FACTORY PATTERN EXTRACTORS (Return dicts for flexible NER)
# ============================================================================


class BaseExtractor:
    """Base class for all NER extractors."""
    
    def __init__(self, language: str = 'en', extractor_type: str = None):
        self.language = language
        self.config = get_config()
        self.extractor_type = extractor_type
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text. Must be implemented by subclasses.
        
        Args:
            text: Input text to process
            
        Returns:
            List of entity dictionaries with keys: text, label, start, end, confidence
        """
        raise NotImplementedError
    
    def _normalize_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize entity: apply label mapping and normalize text/label format.
        
        Args:
            entity: Entity dictionary with 'text', 'label', 'start', 'end', optionally 'confidence'
            
        Returns:
            Normalized entity dictionary (new copy, doesn't mutate input)
        """
        entity = dict(entity)
        
        entity['text'] = entity['text'].strip()
        
        label = entity['label'].upper()
        
        if self.extractor_type and self.extractor_type in LABEL_MAPPINGS:
            mapping = LABEL_MAPPINGS[self.extractor_type]
            entity['label'] = mapping.get(label, label).upper()
        else:
            entity['label'] = label
        
        return entity
    
    def _resolve_overlaps(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve overlapping entities by keeping the one with highest confidence.
        When confidence ties, prefer longer spans.
        Only removes overlapping entities if they have the same label (entity type).
        
        Args:
            entities: List of entities
            
        Returns:
            List of entities (overlapping entities with same label resolved)
        """
        if not entities:
            return []
        
        # Sort by (confidence desc, length desc) - prefer higher confidence, then longer spans
        sorted_entities = sorted(
            entities, 
            key=lambda x: (x.get('confidence', 1.0), x['end'] - x['start']), 
            reverse=True
        )
        
        resolved = []
        for entity in sorted_entities:
            # Check if this entity overlaps with any already resolved entity of the SAME label
            # Using half-open intervals [start, end): ranges overlap if not (end1 <= start2 or end2 <= start1)
            overlaps_same_label = any(
                not (entity['end'] <= existing['start'] or existing['end'] <= entity['start'])
                and entity['label'] == existing['label']
                for existing in resolved
            )
            if not overlaps_same_label:
                resolved.append(entity)
        
        return resolved
    
    def post_process_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Post-process entities: normalize and resolve overlaps.
        
        Normalizes entity text/labels, applies label mappings, and resolves overlapping
        entities with the same label by keeping the highest confidence one.
        
        Args:
            entities: List of raw entity dictionaries
            
        Returns:
            List of normalized and processed entities
        """
        if not self.config.ner.post_process:
            # Still normalize entities even if post-processing is disabled
            return [self._normalize_entity(e) for e in entities]
        
        # Normalize all entities first
        normalized = [self._normalize_entity(e) for e in entities]
        
        # Resolve overlaps (this also handles exact duplicates since they overlap)
        return self._resolve_overlaps(normalized)


class SpacyExtractor(BaseExtractor):
    """Extract entities using spaCy models."""
    
    def __init__(self, language: str = 'en'):
        super().__init__(language, extractor_type='spacy')
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy NER."""
        try:
            nlp = model_manager.get_spacy_model(self.language)
            
            if nlp is None:
                logging.warning(f"spaCy model not available for language '{self.language}'")
                return []
            
            doc = nlp(text)
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.7  # spaCy doesn't provide confidence scores for NER
                })
            
            return self.post_process_entities(entities)
            
        except Exception as e:
            logging.warning(f"Error in spaCy extraction ({self.language}): {e}")
            return []


class HuggingFaceExtractor(BaseExtractor):
    """Extract entities using Hugging Face transformers pipeline."""
    
    def __init__(self, language: str = 'en'):
        super().__init__(language, extractor_type='huggingface')
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using Hugging Face NER pipeline."""
        try:
            ner_pipeline = model_manager.get_huggingface_model(self.language)
            
            if ner_pipeline is None:
                logging.warning(f"HuggingFace model not available for language '{self.language}'")
                return []
            
            results = ner_pipeline(text)
            
            entities = []
            for result in results:
                entities.append({
                    'text': result['word'],
                    'label': result['entity_group'],
                    'start': result['start'],
                    'end': result['end'],
                    'confidence': result.get('score', 1.0)
                })
            
            return self.post_process_entities(entities)
            
        except Exception as e:
            logging.warning(f"Error in HuggingFace extraction ({self.language}): {e}")
            return []


class FlairExtractor(BaseExtractor):
    """Extract entities using Flair models."""
    
    def __init__(self, language: str = 'de', model_name: str = None):
        super().__init__(language, extractor_type='flair')
        self.model_name = model_name  # Optional override, otherwise uses language from config
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using Flair NER."""
        try:
            # Load the Flair SequenceTagger model using language from config or explicit model_name
            tagger = model_manager.get_flair_model(language=self.language, model_name=self.model_name)
            
            if tagger is None:
                logging.warning(f"Flair model not available for language '{self.language}'")
                return []
            
            # Create sentence (don't use tokenizer for legal texts as recommended)
            sentence = Sentence(text, use_tokenizer=False)
            
            # Predict NER tags using the SequenceTagger
            tagger.predict(sentence)
            
            entities = []
            # Iterate over entities and extract information
            for entity in sentence.get_spans('ner'):
                label_obj = entity.get_label('ner')
                entities.append({
                    'text': entity.text,
                    'label': label_obj.value,
                    'start': entity.start_position,
                    'end': entity.end_position,
                    'confidence': label_obj.score
                })
            
            return self.post_process_entities(entities)
            
        except Exception as e:
            model_info = self.model_name or f"config({self.language})"
            logging.warning(f"Error in Flair extraction ({model_info}): {e}")
            return []


class TitleExtractor(BaseExtractor):
    """Extract document title using Hugging Face Gemma model."""
    
    def __init__(self, language: str = 'nl'):
        super().__init__(language, extractor_type='title')
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract title from text using Gemma model.
        
        Returns a single entity with label 'TITLE'.
        """
        try:
            # Load the title extraction pipeline
            generator = model_manager.get_title_extraction_model()
            
            # Prepare the prompt combining instruction and text
            prompt = f"{TITLE_EXTRACTION_INSTRUCTION}\n\nText:\n{text}"
            
            # Create conversation format matching HuggingFace example
            conversation = [{"role": "user", "content": prompt}]
            
            # Generate the title (matching HF example format)
            max_tokens = NER_MODELS['title_extraction']['max_new_tokens']
            output = generator(
                conversation, 
                max_new_tokens=max_tokens, 
                return_full_text=False
            )[0]
            
            # Parse the generated text to extract JSON
            generated_text = output['generated_text']
            
            # Try to extract JSON from the response
            try:
                # Look for JSON in the response
                if '{' in generated_text and '}' in generated_text:
                    start_idx = generated_text.find('{')
                    end_idx = generated_text.rfind('}') + 1
                    json_str = generated_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    title = result.get('title', '').strip()
                else:
                    # Fallback: treat whole response as title
                    title = generated_text.strip()
            except json.JSONDecodeError:
                # If JSON parsing fails, use the whole response
                title = generated_text.strip()
            
            if title:
                # Try to find the title in the original text
                start_pos = text.find(title)
                if start_pos != -1:
                    # Title found in original text
                    entities = [{
                        'text': title,
                        'label': 'TITLE',
                        'start': start_pos,
                        'end': start_pos + len(title),
                        'confidence': 1.0  # LLM doesn't provide confidence, default to 1.0
                    }]
                else:
                    # Title generated/extracted but not exact match in text
                    # Set start=0, end=0 to indicate it's a generated/inferred title
                    entities = [{
                        'text': title,
                        'label': 'TITLE',
                        'start': 0,
                        'end': 0,
                        'confidence': 0.8  # Lower confidence for generated titles
                    }]
                return entities
            
            return []
            
        except Exception as e:
            logging.warning(f"Error in title extraction: {e}")
            return []


class RegexExtractor(BaseExtractor):
    """Extract entities using regex patterns."""
    
    def __init__(self, language: str = 'en', patterns: Dict[str, List[str]] = None):
        super().__init__(language, extractor_type='regex')
        self.patterns = patterns or {}
        self._compiled_patterns = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        for label, pattern_list in self.patterns.items():
            self._compiled_patterns[label] = [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in pattern_list
            ]
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using regex patterns."""
        entities = []
        
        for label, compiled_patterns in self._compiled_patterns.items():
            for pattern in compiled_patterns:
                for match in pattern.finditer(text):
                    entities.append({
                        'text': match.group(0),
                        'label': label,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 1.0  # Regex matches are deterministic
                    })
        
        return self.post_process_entities(entities)


class LanguageRegexExtractor(RegexExtractor):
    """Extract entities using all regex patterns for a specific language."""
    
    def __init__(self, language: str):
        # Get all regex patterns for this language and convert to uppercase labels
        language_patterns = REGEX_PATTERNS.get(language, {})
        patterns = {}
        
        for pattern_type, pattern_list in language_patterns.items():
            # Convert pattern type to uppercase for entity labels (date -> DATE)
            label = pattern_type.upper()
            patterns[label] = pattern_list
        
        super().__init__(language, patterns)


class CompositeExtractor(BaseExtractor):
    """Combine multiple extractors into one unified extractor."""
    
    def __init__(self, extractors: List[BaseExtractor]):
        super().__init__()
        self.extractors = extractors
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using all configured extractors."""
        all_entities = []
        
        for extractor in self.extractors:
            try:
                entities = extractor.extract(text)
                all_entities.extend(entities)
            except Exception as e:
                logging.warning(f"Error in extractor {type(extractor).__name__}: {e}")
                continue
        
        return self.post_process_entities(all_entities)


# Pre-configured extractors for common use cases
def create_german_composite_extractor() -> CompositeExtractor:
    """Create a comprehensive German NER extractor using Flair's legal model."""
    return CompositeExtractor([
        FlairExtractor('de'),  # Uses config: NER_MODELS['flair']['de']
        LanguageRegexExtractor('de')
    ])


def create_dutch_composite_extractor() -> CompositeExtractor:
    """Create a comprehensive Dutch NER extractor."""
    return CompositeExtractor([
        HuggingFaceExtractor('nl'),
        LanguageRegexExtractor('nl')
    ])


def create_english_composite_extractor() -> CompositeExtractor:
    """Create a comprehensive English NER extractor."""
    return CompositeExtractor([
        HuggingFaceExtractor('en'),
        LanguageRegexExtractor('en')  # Will be empty unless patterns are added
    ])
