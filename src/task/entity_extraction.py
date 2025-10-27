from typing import Any
from ..ner_functions import extract_entities
from ..annotation import TripletAnnotation
from ..sparql_config import TASK_OPERATIONS, AI_COMPONENTS, AGENT_TYPES
from .base import DecisionTask
from escape_helpers import sparql_escape_string


class EntityExtractionTask(DecisionTask):
    """Task that extracts named entities from text."""

    __task_type__ = TASK_OPERATIONS["entity_extraction"]

    def create_title_relation(self, source_uri: str, entities: list[dict[str, Any]]):
        for entity in entities:
            if entity['label'] == 'TITLE':
                TripletAnnotation(
                    subject=self.source,
                    predicate="dct:title",
                    obj=sparql_escape_string(entity['text']),
                    activity_id=self.task_uri,
                    source_uri=source_uri,
                    start=entity['start'],
                    end=entity['end'],
                    agent=AI_COMPONENTS["ner_extractor"],
                    agent_type=AGENT_TYPES["ai_component"]
                ).add_to_triplestore()
                self.logger.info(
                    f"Created Title triplet suggestion for '{entity['text']}' ({entity['label']}) at [{entity['start']}:{entity['end']}]")

    def create_en_translation(self, task_data: str) -> str:
        return None

    def extract_general_entities(self, task_data: str, language: str = 'dutch', method: str = 'regex') -> list[
        dict[str, Any]]:
        """
        Extract general NER entities (PERSON, ORG, DATE, etc.) from text.

        Args:
            task_data: Text to extract entities from
            language: Language for extraction ('dutch', 'german', 'english')
            method: Extraction method ('regex', 'spacy', 'flair', 'composite', 'title')
        """
        self.logger.info(f"Extracting general entities using {method}/{language}")

        # Extract entities using the factory pattern
        entities = extract_entities(task_data, language=language, method=method)
        self.logger.info(f"Found {len(entities)} general entities")

        return entities

    def process(self):
        eli_expression = self.fetch_data()
        self.logger.info(eli_expression)

        # Uses defaults from ner_config.py: language='dutch', method='regex'
        # Language can be passed in future when extracted from database
        # todo fallback to source to be removed
        uri_of_translation_expr = self.create_en_translation(eli_expression) or self.source
        entities = self.extract_general_entities(eli_expression)

        # todo to be improved upon a lot by inserting functions to create the other predicates
        # ELI properties
        self.create_title_relation(uri_of_translation_expr, entities)