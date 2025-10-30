import contextlib
import logging
import os
import json
import langdetect
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
from .classifier.train import train


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

    def create_language_relation(self, source_uri: str, language: str):
        language_mapping = {
            'nl': "http://publications.europa.eu/resource/authority/language/NLD",
            'de': "http://publications.europa.eu/resource/authority/language/DEU",
            'en': "http://publications.europa.eu/resource/authority/language/ENG"
        }
        TripletAnnotation(
            subject=self.source,
            predicate="eli:language",
            obj=sparql_escape_uri(language_mapping.get(language)),
            activity_id=self.task_uri,
            source_uri=source_uri,
            start=0,
            end=0,
            agent=AI_COMPONENTS["ner_extractor"],
            agent_type=AGENT_TYPES["ai_component"]
        ).add_to_triplestore()

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
        # todo fallback to source to be removed
        return self.source

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

        language = langdetect.detect(eli_expression)
        # todo replace first argument with eli:expression uri
        self.create_language_relation(self.source, language)

        # Uses defaults from ner_config.py: language='dutch', method='regex'
        # Language can be passed in future when extracted from database
        uri_of_translation_expr = self.create_en_translation(eli_expression)
        #entities = self.extract_general_entities(eli_expression, language)

        # todo to be improved upon a lot by inserting functions to create the other predicates
        # ELI properties
        entities = extract_entities(eli_expression, language=language, method='title')
        self.create_title_relation(uri_of_translation_expr, entities)


class ModelAnnotatingTask(DecisionTask):
    """Task that links the correct code from a list to text."""
    
    __task_type__ = TASK_OPERATIONS["model_annotation"]

    def __init__(self, task_uri: str, source: str | None = None):
        if source is None:
            super().__init__(task_uri)
        else:
            self.task_uri = task_uri
            self.logger = logging.getLogger(self.__class__.__name__)
            self.source = source

        self._llm_config = {
            "model_name": os.getenv("LLM_MODEL_NAME"),
            "temperature": float(os.getenv("LLM_TEMPERATURE")),
        }

        self._llm_system_message = "You are a juridical and administrative assistant that must determine the best matching codes from a list with a given text."
        self._llm_user_message = "Determine the best matching codes from the following list for the given public decision.\n\n" \
            "\"\"\"" \
            "CODE LIST:\n" \
            "{code_list}\n" \
            "\"\"\"\n\n" \
            "\"\"\"" \
            "DECISION TEXT:\n" \
            "{decision_text}\n" \
            "\"\"\"" \
            "Provide your answer as a list of strings representing the matching codes. Provide all matching codes (can be a single one), but only those that are truly matching and only from the given list!"

        self._llm = OpenAIModel(self._llm_config)

    def process(self):
        task_data = self.fetch_data()
        self.logger.info(task_data)

        # TO DO: ADD FUNCTION TO RETRIEVE ACTUAL CODE LIST
        sdgs = ["SDG-01 No Poverty",
                "SDG-02 Zero Hunger",
                "SDG-03 Good Health and Well-Being",
                "SDG-04 Quality Education",
                "SDG-05 Gender Equality",
                "SDG-06 Clean Water and Sanitation",
                "SDG-07 Affordable and Clean Energy",
                "SDG-08 Decent Work and Economic Growth",
                "SDG-09 Industry, Innovation and Infrastructure",
                "SDG-10 Reduced Inequality",
                "SDG-11 Sustainable Cities and Communities",
                "SDG-12 Responsible Consumption and Production",
                "SDG-13 Climate Action",
                "SDG-14 Life Below Water",
                "SDG-16 Life on Land",
                "SDG-16 Peace, Justice and Strong Institutions",
                "SDG-17 Partnerships for the Goals"
                ]

        llm_input = LlmTaskInput(system_message=self._llm_system_message,
                                 user_message=self._llm_user_message.format(
                                     code_list=sdgs, decision_text=task_data),
                                 assistant_message=None,
                                 output_format=EntityLinkingTaskOutput)

        response = self._llm(llm_input)
        classes = [designated_class.replace(" ", "_") for designated_class in response.designated_classes]

        for c in classes:
            annotation = LinkingAnnotation(
                self.task_uri,
                self.source,
                # TO DO: CHANGE TO ACTUAL URI
                "http://example.org/" + c,
                AI_COMPONENTS["model_annotater"],
                AGENT_TYPES["ai_component"]
            )
            annotation.add_to_triplestore()


class ModelBatchAnnotatingTask(Task, ABC):
    """Task that creates ModelAnnotatingTasks for all decisions that are not yet annotated."""

    __task_type__ = TASK_OPERATIONS["model_batch_annotation"]

    def __init__(self, task_uri: str):
        super().__init__(task_uri)

    def process(self):
        decision_uris = self.fetch_decisions_without_annotations()
        print(f"{len(decision_uris)} decisions to process.", flush=True)

        for i, decision_uri in enumerate(decision_uris):
            ModelAnnotatingTask(self.task_uri, decision_uri).process()
            print(f"Processed decision {i+1}/{len(decision_uris)}: {decision_uri}", flush=True)

    def fetch_decisions_without_annotations(self) -> list[str]:
        q = get_prefixes_for_query("rdf", "eli", "oa") + \
            """
            SELECT DISTINCT ?s
            WHERE {
                GRAPH <http://mu.semte.ch/graphs/oslo-temp> {
                    ?s rdf:type eli:Expression .
                }
                FILTER NOT EXISTS {
                    GRAPH <http://mu.semte.ch/graphs/ai> {
                    ?ann a oa:Annotation ;
                        oa:hasTarget ?s ;
                        oa:motivatedBy oa:classifying .
                    }
                }
            }
            """
        
        response = query(q)
        bindings = response.get("results", {}).get("bindings", [])
        decision_uris = [b["s"]["value"] for b in bindings if "s" in b]

        return decision_uris
    

class ClassifierTrainingTask(Task, ABC):
    """Task that trains a classifier for the available annotations in the triple store."""

    __task_type__ = TASK_OPERATIONS["classifier_training"]

    def __init__(self, task_uri: str):
        super().__init__(task_uri)

    def process(self):
        decisions = self.fetch_decisions_with_classes()
        decisions = self.convert_classes_to_original_names(decisions)

        # TO DO: ADD FUNCTION TO RETRIEVE ACTUAL CODE LIST
        sdgs = ["SDG-01 No Poverty",
                "SDG-02 Zero Hunger",
                "SDG-03 Good Health and Well-Being",
                "SDG-04 Quality Education",
                "SDG-05 Gender Equality",
                "SDG-06 Clean Water and Sanitation",
                "SDG-07 Affordable and Clean Energy",
                "SDG-08 Decent Work and Economic Growth",
                "SDG-09 Industry, Innovation and Infrastructure",
                "SDG-10 Reduced Inequality",
                "SDG-11 Sustainable Cities and Communities",
                "SDG-12 Responsible Consumption and Production",
                "SDG-13 Climate Action",
                "SDG-14 Life Below Water",
                "SDG-16 Life on Land",
                "SDG-16 Peace, Justice and Strong Institutions",
                "SDG-17 Partnerships for the Goals"
                ]
        
        kwargs = {}

        if os.getenv("TRANSFORMER_NAME"):
            kwargs["transformer"] = os.getenv("TRANSFORMER_NAME")

        if os.getenv("LEARNING_RATE"):
            kwargs["learning_rate"] = float(os.getenv("LEARNING_RATE"))

        if os.getenv("EPOCHS"):
            kwargs["epochs"] = int(os.getenv("EPOCHS"))

        if os.getenv("WEIGHT_DECAY"):
            kwargs["weight_decay"] = float(os.getenv("WEIGHT_DECAY"))

        print("Started training...", flush=True)
        train(decisions[:10],
              sdgs, 
              os.getenv("HUGGINGFACE_OUTPUT_MODEL_ID"),
              **kwargs) 
        print("Done training!", flush=True)

    def convert_classes_to_original_names(self, decisions: list[dict[str, str | list[str]]]):
        for decision in decisions:
            decision["classes"] = [c.split("/")[-1].replace("_", " ") for c in decision["classes"]]

        return decisions

        
    def fetch_decisions_with_classes(self) -> list[dict[str, str | list[str]]]:
        q = get_prefixes_for_query("rdf", "eli", "oa") + \
        """
        SELECT ?decision ?title ?description ?decision_basis ?classes
        WHERE {
        {
            SELECT ?decision (GROUP_CONCAT(DISTINCT STR(?body); separator="|") AS ?classes)
            WHERE {
                GRAPH <http://mu.semte.ch/graphs/ai> {
                    ?ann a oa:Annotation ;
                        oa:hasTarget ?decision ;
                        oa:motivatedBy oa:classifying ;
                        oa:hasBody ?body .
                }
            }
            GROUP BY ?decision
        }
            GRAPH <http://mu.semte.ch/graphs/oslo-temp> {
                ?decision rdf:type eli:Expression .
                OPTIONAL { ?decision eli:title ?title }
                OPTIONAL { ?decision eli:description ?description }
                OPTIONAL { ?decision <http://data.europa.eu/eli/eli-dl#decision_basis> ?decision_basis }
            }
        }
        """

        res = query(q)
        bindings = res.get("results", {}).get("bindings", [])

        results = []
        for b in bindings:
            decision = b["decision"]["value"]
            classes_concat = b.get("classes", {}).get("value", "")
            classes = [c for c in classes_concat.split("|") if c]
            title = b.get("title", {}).get("value", "")
            description = b.get("description", {}).get("value", "")
            decision_basis = b.get("decision_basis", {}).get("value", "")

            text = "\n".join([t for t in [title, description, decision_basis] if t])

            results.append({
                "decision": decision,
                "classes": classes,
                "text": text
            })

        return results
    
