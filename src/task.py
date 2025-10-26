import contextlib
import logging
import os
import json
from abc import ABC, abstractmethod
from typing import Optional, Any

from uuid import uuid4
from string import Template
from helpers import query
from escape_helpers import sparql_escape_uri, sparql_escape_string

from .helper_functions import clean_string, get_start_end_offsets, process_text, geocode_detectable
from .ner_extractors import SpacyGeoAnalyzer
from .ner_functions import extract_entities
from .nominatim_geocoder import NominatimGeocoder
from .annotation import GeoAnnotation, LinkingAnnotation, TripletAnnotation
from .sparql_config import get_prefixes_for_query, GRAPHS, JOB_STATUSES, TASK_OPERATIONS, AI_COMPONENTS, AGENT_TYPES
from .llm_models.llm_model_clients import OpenAIModel
from .llm_models.llm_task_models import LlmTaskInput, EntityLinkingTaskOutput


class Task(ABC):
    """Base class for background tasks that process data from the triplestore."""

    def __init__(self, task_uri: str):
        super().__init__()
        self.task_uri = task_uri
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def lookup(cls, task_type: str) -> Optional['Task']:
        """
        Yield all subclasses of the given class, per:
        https://adamj.eu/tech/2024/05/10/python-all-subclasses/
        """
        for subclass in cls.__subclasses__():
            if hasattr(subclass, '__task_type__') and subclass.__task_type__ == task_type:
                return subclass
            else:
                res = subclass.lookup(task_type)
                if res is not None:
                    return res
        return None

    @classmethod
    def from_uri(cls, task_uri: str) -> 'Task':
        """Create a Task instance from its URI in the triplestore."""
        q = Template(
            get_prefixes_for_query("adms", "task") +
            """
            SELECT ?task ?taskType WHERE {
              ?task task:operation ?taskType .
              FILTER(?task = $uri)
            }
        """).substitute(uri=sparql_escape_uri(task_uri))
        for b in query(q).get('results').get('bindings'):
            candidate_cls = cls.lookup(b['taskType']['value'])
            if candidate_cls is not None:
                return candidate_cls(task_uri)
            raise RuntimeError("Unknown task type {0}".format(b['taskType']['value']))
        raise RuntimeError("Task with uri {0} not found".format(task_uri))

    def change_state(self, old_state: str, new_state: str, results_container_uri: str = "") -> None:
        """Update the task status in the triplestore."""
        query_template = Template(
            get_prefixes_for_query("task", "adms") +
            """
            DELETE {
            GRAPH <""" + GRAPHS["jobs"] + """> {
                ?task adms:status ?oldStatus .
            }
            }
            INSERT {
            GRAPH <""" + GRAPHS["jobs"] + """> {
                ?task
                $results_container_line
                adms:status <$new_status> .

            }
            }
            WHERE {
            GRAPH <""" + GRAPHS["jobs"] + """> {
                BIND($task AS ?task)
                BIND(<$old_status> AS ?oldStatus)
                OPTIONAL { ?task adms:status ?oldStatus . }
            }
            }
            """)

        results_container_line = ""
        if results_container_uri:
            results_container_line = f"task:resultsContainer <{results_container_uri}> ;"

        query_string = query_template.substitute(
            new_status=JOB_STATUSES[new_state],
            old_status=JOB_STATUSES[old_state],
            task=sparql_escape_uri(self.task_uri),
            results_container_line=results_container_line)

        query(query_string)

    @contextlib.contextmanager
    def run(self):
        """Context manager for task execution with state transitions."""
        self.change_state("scheduled", "busy")
        yield
        self.change_state("busy", "success")

    def execute(self):
        """Run the task and handle state transitions."""
        with self.run():
            self.process()

    @abstractmethod
    def process(self):
        """Process task data (implemented by subclasses)."""
        pass


class DecisionTask(Task, ABC):
    """Task that processes decision-making data with input and output containers."""
    
    def __init__(self, task_uri: str):
        super().__init__(task_uri)

        q = Template(
            get_prefixes_for_query("dct", "task", "nfo") +
            """
        SELECT ?source WHERE {
          BIND($task AS ?t)
          ?t a task:Task .
          OPTIONAL { 
            ?t task:inputContainer ?ic . 
            OPTIONAL { ?ic a nfo:DataContainer ; task:hasResource ?source . }
          }
        }
        """).substitute(task=sparql_escape_uri(task_uri))
        r = query(q)
        self.source = r["results"]["bindings"][0]["source"]["value"]

    def fetch_data(self) -> str:
        """Retrieve the input data for this task from the triplestore."""
        query_string = f"""
        SELECT ?title ?description ?decision_basis WHERE {{
        BIND(<{self.source}> AS ?s)
        OPTIONAL {{ ?s <http://data.europa.eu/eli/ontology#title> ?title }}
        OPTIONAL {{ ?s <http://data.europa.eu/eli/ontology#description> ?description }}
        OPTIONAL {{ ?s <http://data.europa.eu/eli/eli-dl#decision_basis> ?decision_basis }}
        }}
        """

        query_result = query(query_string)

        title = query_result["results"]["bindings"][0]["title"]["value"]
        description = query_result["results"]["bindings"][0]["description"]["value"]
        decision_basis = query_result["results"]["bindings"][0]["decision_basis"]["value"]

        return "\n".join([title, description, decision_basis])


class GeoExtractionTask(DecisionTask):
    """Task that geocodes location information from text."""

    __task_type__ = TASK_OPERATIONS["geo_extraction"]

    ner_analyzer = SpacyGeoAnalyzer(model_path=os.getenv("NER_MODEL_PATH"), labels=json.loads(os.getenv("NER_LABELS")))
    geocoder = NominatimGeocoder(base_url=os.getenv("NOMINATIM_BASE_URL"), rate_limit=0.5)

    def apply_geo_entities(self, task_data: str):
        """Extract geographic entities from text and store as annotations."""
        default_city = "Gent"

        cleaned_text = clean_string(task_data)
        detectables, _, doc = process_text(cleaned_text, self.__class__.ner_analyzer, default_city)

        if hasattr(doc, 'error'):
            self.logger.error(f"Error: {doc['error']}")
        else:
            # Geocoding Results
            if detectables:
                self.logger.info("Geocoding Results")

                for geo_entity in ["streets", "addresses"]:
                    if geo_entity in detectables:
                        for detectable in detectables[geo_entity]:
                            result = geocode_detectable(detectable, self.__class__.geocoder, default_city)
                            print(result)

                            if result["success"]:
                                if geo_entity == "streets":
                                    offsets = get_start_end_offsets(task_data, detectable["name"])
                                    start_offset = offsets[0][0]
                                    end_offset = offsets[0][1]
                                    annotation = GeoAnnotation(
                                        result.get("geojson", {}),
                                        self.task_uri,
                                        self.source,
                                        "http://example.org/{0}".format(uuid4()),
                                        start_offset,
                                        end_offset,
                                        AI_COMPONENTS["ner_extractor"],
                                        AGENT_TYPES["ai_component"]
                                    )
                                    annotation.add_to_triplestore()
                            self.logger.info(result)
            else:
                self.logger.info("No location entities detected.")

    def process(self):
        task_data = self.fetch_data()
        self.logger.info(task_data)
        self.apply_geo_entities(task_data)
        
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
                self.logger.info(f"Created Title triplet suggestion for '{entity['text']}' ({entity['label']}) at [{entity['start']}:{entity['end']}]")

    def create_en_translation(self, task_data: str) -> str:
        return None

    def extract_general_entities(self, task_data: str, language: str = 'dutch', method: str = 'regex') -> list[dict[str, Any]]:
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


class EntityLinkingTask(DecisionTask):
    """Task that links the correct code from a list to text."""
    
    __task_type__ = TASK_OPERATIONS["entity_linking"]

    def __init__(self, task_uri: str):
        super().__init__(task_uri)

        self._llm_config = {
            "model_name": os.getenv("LLM_MODEL_NAME"),
            "temperature": float(os.getenv("LLM_TEMPERATURE")),
        }

        self._llm_system_message = "You are a juridical and administrative assistant that must determine the best matching code from a list with a given text."
        self._llm_user_message = "Determine the best matching code from the following list for the given public decision.\n\n" \
            "\"\"\"" \
            "CODE LIST:\n" \
            "{code_list}\n" \
            "\"\"\"\n\n" \
            "\"\"\"" \
            "DECISION TEXT:\n" \
            "{decision_text}\n" \
            "\"\"\"" \
            "Answer only and solely with one of the given codes!"

        self._llm = OpenAIModel(self._llm_config)

    def process(self):
        task_data = self.fetch_data()
        self.logger.info(task_data)

        # TO DO: ADD FUNCTION TO RETRIEVE ACTUAL CODE LIST
        sdgs = ["No Poverty",
                "Zero Hunger",
                "Good Health and Well-Being",
                "Quality Education",
                "Gender Equality",
                "Clean Water and Sanitation",
                "Affordable and Clean Energy",
                "Decent Work and Economic Growth",
                "Industry, Innovation and Infrastructure",
                "Reduced Inequalities",
                "Sustainable Cities and Communities",
                "Responsible Consumption and Production",
                "Climate Action",
                "Life Below Water",
                "Life on Land",
                "Peace, Justice and Strong Institutions",
                "Partnerships for the Goals"
                ]

        llm_input = LlmTaskInput(system_message=self._llm_system_message,
                                 user_message=self._llm_user_message.format(
                                     code_list=sdgs, decision_text=task_data),
                                 assistant_message=None,
                                 output_format=EntityLinkingTaskOutput)

        response = self._llm(llm_input)

        annotation = LinkingAnnotation(
            self.task_uri,
            self.source,
            # TO DO: CHANGE TO ACTUAL URI
            "http://example.org/" +
            response.designated_class.replace(" ", "_"),
            AI_COMPONENTS["entity_linker"],
            AGENT_TYPES["ai_component"]
        )
        annotation.add_to_triplestore()
