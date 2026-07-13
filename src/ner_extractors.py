"""
NER Entity Extractors

This module contains different NER extraction methods organized by approach.

Classes:
- SpacyGeoAnalyzer: Belgian location extraction (returns spaCy Doc)
- BaseExtractor: Base class for factory pattern extractors (returns dicts)
- SpacyExtractor, FlairExtractor, etc.: Factory pattern implementations
"""
import re
from helpers import logger
from flair.data import Sentence
from typing import List, Dict, Any
from .ner_models import model_manager
from .ner_config import REGEX_PATTERNS, NER_MODELS, LABEL_MAPPINGS
from .config import get_config
from .helper_functions import fail_if_no_successes
import torch

# Person-type labels produced by the extractors (HuggingFace -> MANDATARY, spaCy/Flair PER -> PERSON). Validation only applies to these; labels like DATE or LOCATION legitimately contain digits and must be left untouched.
PERSON_LABELS = frozenset({"PERSON", "PER", "MANDATARY"})

# Short digits are legitimate (e.g. "2nd Deputy Mayor"). so allow for some numbers
_LONG_DIGIT_RUN = re.compile(r"\d{3,}")

# Characters that never occur in a person/role span.
_INVALID_NAME_CHARS = frozenset("@#_=|<>{}~^")


def is_valid_person_name(text: str) -> bool:
    """Return False for spans that cannot be a real person name.

    Rejects spans that are empty, contain a long digit run, contain a garbage character, 
    or contain no letters at all. Role-annotation punctuation is allowed.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if _LONG_DIGIT_RUN.search(stripped):
        return False
    if any(ch in _INVALID_NAME_CHARS for ch in stripped):
        return False
    return any(ch.isalpha() for ch in stripped)


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
        # Normalize all entities first
        normalized = [self._normalize_entity(e) for e in entities]

        # Drop malformed person/mandatary spans (digits / invalid characters).
        # Independent of `post_process` so it also runs when overlap resolution is disabled.
        if self.config.ner.validate_persons:
            normalized = self._filter_invalid_persons(normalized)

        # Drop entities below the configured confidence floor. No-op when `min_confidence` is unset/null.
        if self.config.ner.min_confidence is not None:
            normalized = self._filter_by_confidence(normalized, self.config.ner.min_confidence)

        if not self.config.ner.post_process:
            return normalized

        # Resolve overlaps (this also handles exact duplicates since they overlap)
        return self._resolve_overlaps(normalized)

    def _filter_invalid_persons(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop person-type entities whose text cannot be a real name."""
        kept = []
        for entity in entities:
            if entity['label'] in PERSON_LABELS and not is_valid_person_name(entity['text']):
                logger.info(
                    f"Dropping malformed {entity['label']} entity: {entity['text']!r}"
                )
                continue
            kept.append(entity)
        return kept

    def _filter_by_confidence(self, entities: List[Dict[str, Any]], min_confidence: float) -> List[Dict[str, Any]]:
        """Drop entities scoring below ``min_confidence``.

        Entities without a confidence score are treated as 1.0 (kept),
        so deterministic sources (regex) are never affected.
        """
        kept = []
        for entity in entities:
            if entity.get('confidence', 1.0) < min_confidence:
                logger.info(
                    f"Dropping low-confidence {entity['label']} entity "
                    f"{entity['text']!r} (score={entity.get('confidence')})"
                )
                continue
            kept.append(entity)
        return kept


class SpacyExtractor(BaseExtractor):
    """Extract entities using spaCy models."""
    
    def __init__(self, language: str = 'en'):
        super().__init__(language, extractor_type='spacy')
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy NER.

        Raises:
            ValueError / RuntimeError: Propagated from ``ModelManager.get_spacy_model``
                when the spaCy model is not configured or fails to load.
            Exception: Any error from ``nlp(text)`` is propagated so the caller
                (typically ``CompositeExtractor``) can decide whether the failure
                is fatal.
        """
        nlp = model_manager.get_spacy_model(self.language)

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


class HuggingFaceExtractor(BaseExtractor):
    """Extract entities using Hugging Face transformers pipeline."""
    
    def __init__(self, language: str = 'en'):
        super().__init__(language, extractor_type='huggingface')
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using Hugging Face NER pipeline.

        Raises:
            ValueError / RuntimeError: Propagated from ``ModelManager.get_huggingface_model``
                when the model is not configured or fails to load.
            Exception: Inference errors are propagated to the caller.
        """
        ner_pipeline = model_manager.get_huggingface_model(self.language)

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


class FlairExtractor(BaseExtractor):
    """Extract entities using Flair models."""
    
    def __init__(self, language: str = 'de', model_name: str = None):
        super().__init__(language, extractor_type='flair')
        self.model_name = model_name  # Optional override, otherwise uses language from config
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using Flair NER.

        Raises:
            ValueError / RuntimeError: Propagated from ``ModelManager.get_flair_model``
                when the model is not configured or fails to load.
            Exception: Prediction errors are propagated to the caller.
        """
        # Load the Flair SequenceTagger model using language from config or explicit model_name
        tagger = model_manager.get_flair_model(language=self.language, model_name=self.model_name)

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
        """Extract entities using all configured extractors.

        Individual sub-extractor failures are tolerated (other extractors
        may still produce results), but if *every* sub-extractor raises
        the composite raises ``RuntimeError`` so the surrounding task is
        marked as failed.
        """
        all_entities = []
        errors: List[str] = []
        successes = 0

        for extractor in self.extractors:
            extractor_name = type(extractor).__name__
            try:
                entities = extractor.extract(text)
                all_entities.extend(entities)
                successes += 1
            except Exception as e:
                logger.exception(f"Error in extractor {extractor_name}")
                errors.append(f"{extractor_name}: {type(e).__name__}: {e}")

        fail_if_no_successes(
            label=f"CompositeExtractor({self.language or 'unknown'})",
            total=len(self.extractors),
            successes=successes,
            errors=errors,
        )

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

class EntityRefiner:
    """
    Refines generic entity labels to specific types using a Longformer classifier.
    
    This is a standalone post-processing step that can be applied after any NER extraction.
    It refines labels like DATE -> publication_date, LOCATION -> impact_location, etc.
    
    Usage:
        extractor = create_german_composite_extractor()
        entities = extractor.extract(text)
        
        refiner = EntityRefiner()
        refined_entities = refiner.refine(entities, text)
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.logger = logger
    
    def _load_model(self):
        """Lazy load the refinement model."""
        if self.model is None:
            self.model, self.tokenizer = model_manager.get_refinement_model()
    
    def refine(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """
        Refine generic entity labels to specific types.
        
        Args:
            entities: List of entity dictionaries from NER extraction
            text: The original text (needed for context)
            
        Returns:
            List of entities with refined labels. Non-refinable entities are passed through unchanged.
            Refined entities have 'original_label' field preserving the generic label.

        Raises:
            RuntimeError: Propagated from ``ModelManager.get_refinement_model`` if
                the refinement model cannot be loaded. Also raised if every
                refinable entity in ``entities`` fails during refinement (the
                surrounding task is then marked as failed by ``Task.run``).
        """
        self._load_model()

        refinement_config = NER_MODELS.get('refinement', {})
        refinable_labels = set(refinement_config.get('refinable_labels', []))
        label_classes = refinement_config.get('label_classes', [])
        max_length = refinement_config.get('max_length', 2048)
        
        refined_entities = []
        attempted = 0
        successes = 0
        errors: List[str] = []
        for entity in entities:
            # Only refine entities with refinable labels
            if entity['label'] not in refinable_labels:
                refined_entities.append(entity)
                continue

            attempted += 1
            try:
                # Mark entity in context with [E] ... [/E] markers as per model documentation
                entity_text = entity['text']
                start_pos = entity['start']
                end_pos = entity['end']
                
                # Create marked text with entity markers
                marked_text = (
                    text[:start_pos] + 
                    f"[E] {entity_text} [/E]" + 
                    text[end_pos:]
                )
                
                # Tokenize and predict
                inputs = self.tokenizer(
                    marked_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=max_length, 
                    padding="max_length"
                )
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Get predicted class
                pred_idx = torch.argmax(outputs.logits, dim=-1).item()
                
                # Map prediction to label
                if pred_idx < len(label_classes):
                    refined_label = label_classes[pred_idx].upper()
                    
                    # Create refined entity (preserve original label)
                    refined_entity = dict(entity)
                    refined_entity['original_label'] = entity['label']
                    refined_entity['label'] = refined_label
                    refined_entities.append(refined_entity)
                else:
                    # If prediction is out of range, keep original
                    refined_entities.append(entity)
                successes += 1
            except Exception as e:
                self.logger.exception(f"Error refining entity '{entity['text']}'")
                errors.append(f"{entity.get('text')!r}: {type(e).__name__}: {e}")
                refined_entities.append(entity)

        fail_if_no_successes(
            label="EntityRefiner.refine",
            total=attempted,
            successes=successes,
            errors=errors,
        )

        return refined_entities


# Singleton instance for convenience
entity_refiner = EntityRefiner()
