import contextlib
import logging
import os
import importlib
import uuid
import langdetect
import random
from abc import ABC, abstractmethod
from typing import Optional, Any, Type

from uuid import uuid4
from helpers import query, update
from string import Template
from translatepy import Translator
from escape_helpers import sparql_escape_uri, sparql_escape_string

from .helper_functions import clean_string, get_start_end_offsets, process_text, geocode_detectable
from .ner_extractors import SpacyGeoAnalyzer
from .ner_functions import extract_entities
from .config import get_config
from .nominatim_geocoder import NominatimGeocoder
from .annotation import GeoAnnotation, LinkingAnnotation, TripletAnnotation, NERAnnotation
from .sparql_config import get_prefixes_for_query, GRAPHS, JOB_STATUSES, TASK_OPERATIONS, AI_COMPONENTS, AGENT_TYPES, LANGUAGE_CODE_TO_URI, LANGUAGE_URI_TO_CODE
from .llm_models.llm_model_clients import OpenAIModel
from .llm_models.llm_task_models import LlmTaskInput, EntityLinkingTaskOutput
from .classifier.train import train


class Task(ABC):
    """Base class for background tasks that process data from the triplestore."""

    def __init__(self, task_uri: str):
        super().__init__()
        self.task_uri = task_uri
        self.results_container_uris = []
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def supported_operations(cls) -> list[Type['Task']]:
        all_ops = []
        for subclass in cls.__subclasses__():
            if hasattr(subclass, '__task_type__'):
                all_ops.append(subclass)
            else:
                all_ops.extend(subclass.supported_operations())
        return all_ops

    @classmethod
    def lookup(cls, task_type: str) -> Optional['Task']:
        """
        Yield all subclasses of the given class, per:
        """
        for subclass in cls.supported_operations():
            if hasattr(subclass, '__task_type__') and subclass.__task_type__ == task_type:
                return subclass
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
        for b in query(q, sudo=True).get('results').get('bindings'):
            candidate_cls = cls.lookup(b['taskType']['value'])
            if candidate_cls is not None:
                return candidate_cls(task_uri)
            raise RuntimeError(
                "Unknown task type {0}".format(b['taskType']['value']))
        raise RuntimeError("Task with uri {0} not found".format(task_uri))

    def change_state(self, old_state: str, new_state: str) -> None:
        """Update the task status in the triplestore."""

        # Update the task status
        status_query = Template(
            get_prefixes_for_query("task", "adms") +
            """
            DELETE {
            GRAPH <""" + GRAPHS["jobs"] + """> {
                ?task adms:status ?oldStatus .
            }
            }
            INSERT {
            GRAPH <""" + GRAPHS["jobs"] + """> {
                ?task adms:status <$new_status> .
            }
            }
            WHERE {
            GRAPH <""" + GRAPHS["jobs"] + """> {
                BIND($task AS ?task)
                BIND(<$old_status> AS ?oldStatus)
                OPTIONAL { ?task adms:status ?oldStatus . }
            }
            }
            """
        )
        query_string = status_query.substitute(
            new_status=JOB_STATUSES[new_state],
            old_status=JOB_STATUSES[old_state],
            task=sparql_escape_uri(self.task_uri)
        )

        update(query_string, sudo=True)

        # Batch-insert results containers (if any)
        if self.results_container_uris:
            BATCH_SIZE = 50
            insert_template = Template(
                get_prefixes_for_query("task", "adms") +
                """
                INSERT {
                GRAPH <""" + GRAPHS["jobs"] + """> {
                    ?task $results_container_line .
                }
                }
                WHERE {
                    BIND($task AS ?task)
                }
                """
            )

            for i in range(0, len(self.results_container_uris), BATCH_SIZE):
                batch_uris = self.results_container_uris[i:i + BATCH_SIZE]
                results_container_line = " ;\n".join(
                    [f"task:resultsContainer {sparql_escape_uri(uri)}" for uri in batch_uris]
                )
                query_string = insert_template.substitute(
                    task=sparql_escape_uri(self.task_uri),
                    results_container_line=results_container_line
                )
                update(query_string, sudo=True)

    @contextlib.contextmanager
    def run(self):
        """Context manager for task execution with state transitions."""
        self.change_state("scheduled", "busy")
        try:
            yield
            self.change_state("busy", "success")
        except Exception as e:
            self.logger.error(
                f"Task {self.task_uri} failed: {type(e).__name__}: {str(e)}", exc_info=True)
            try:
                self.change_state("busy", "failed")
            except Exception as state_error:
                self.logger.error(
                    f"Failed to update task {self.task_uri} status to failed: {state_error}")
            raise

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
        self.source_graph: Optional[str] = None

        q = Template(
            get_prefixes_for_query("dct", "task", "nfo") +
            """
        SELECT ?source WHERE {
          VALUES ?t {
            $task
          }
          ?t a task:Task .
          OPTIONAL { 
            ?t task:inputContainer ?ic . 
            OPTIONAL { ?ic a nfo:DataContainer ; task:hasResource ?source . }
          }
        }
        """).substitute(task=sparql_escape_uri(task_uri))
        r = query(q, sudo=True)
        bindings = r.get("results", {}).get("bindings", [])
        if not bindings or "source" not in bindings[0] or "value" not in bindings[0].get("source", {}):
            raise ValueError(f"No source found for task {task_uri}")
        self.source = bindings[0]["source"]["value"]

    def fetch_data(self) -> str:
        """Retrieve the input data for this task from the triplestore."""
        query_template = Template(
            get_prefixes_for_query("eli", "eli-dl", "dct", "epvoc") +
            """
            SELECT DISTINCT ?graph ?title ?description ?decision_basis ?content ?lang
            WHERE {
              GRAPH ?graph {
                VALUES ?s {
                  $source
                }
                ?s a ?thing .
                OPTIONAL { ?s eli:title ?title }
                OPTIONAL { ?s eli:description ?description }
                OPTIONAL { ?s eli-dl:decision_basis ?decision_basis }
                OPTIONAL { ?s epvoc:expressionContent ?content }
                OPTIONAL { ?s dct:language ?lang }
              }
            }
        """)

        query_result = query(query_template.substitute(
            source=sparql_escape_uri(self.source)
        ), sudo=True)

        bindings = query_result.get("results", {}).get("bindings", [])
        texts: list[str] = []
        seen = set()
        for binding in bindings:
            # Cache the graph of the source expression so we can reuse it later
            if not self.source_graph:
                self.source_graph = binding.get("graph", {}).get("value")
            for field in ("content", "title", "description", "decision_basis"):
                value = binding.get(field, {}).get("value")
                if value and value not in seen:
                    texts.append(value)
                    seen.add(value)

        return "\n".join(texts)

    def fetch_work_uri(self) -> Optional[str]:
        """
        Retrieve the eli:work realized by this expression, if available.
        """
        query_template = Template(
            get_prefixes_for_query("eli") +
            """
            SELECT ?work WHERE {
              GRAPH ?g {
                $source eli:realizes ?work .
              }
            }
            LIMIT 1
            """
        )

        query_result = query(
            query_template.substitute(source=sparql_escape_uri(self.source)),
            sudo=True
        )
        bindings = query_result.get("results", {}).get("bindings", [])
        if bindings and "work" in bindings[0]:
            work_uri = bindings[0]["work"]["value"]
            self.logger.info(
                f"Found work {work_uri} for expression {self.source}")
            return work_uri

        self.logger.warning(
            f"No eli:realizes work found for expression {self.source}")
        return None


class GeoExtractionTask(DecisionTask):
    """Task that geocodes location information from text."""

    __task_type__ = TASK_OPERATIONS["geo_extraction"]
    config = get_config()
    ner_analyzer = SpacyGeoAnalyzer(model_path=os.getenv(
        "NER_MODEL_PATH"), labels=config.ner.labels)
    geocoder = NominatimGeocoder(base_url=str(
        config.geocoding.nominatim_base_url), rate_limit=0.5)

    def apply_geo_entities(self, task_data: str):
        """Extract geographic entities from text and store as annotations."""
        default_city = "Gent"

        cleaned_text = clean_string(task_data)
        detectables, _, doc = process_text(
            cleaned_text, self.__class__.ner_analyzer, default_city)

        if hasattr(doc, 'error'):
            self.logger.error(f"Error: {doc['error']}")
        else:
            if detectables:
                self.logger.info("Geocoding Results")

                for geo_entity in ["streets", "addresses"]:
                    if geo_entity in detectables:
                        for detectable in detectables[geo_entity]:
                            result = geocode_detectable(
                                detectable, self.__class__.geocoder, default_city)
                            print(result)

                            if result["success"]:
                                if geo_entity == "streets":
                                    offsets = get_start_end_offsets(
                                        task_data, detectable["name"])
                                    start_offset = offsets[0][0]
                                    end_offset = offsets[0][1]
                                    annotation = GeoAnnotation(
                                        result.get("geojson", {}),
                                        self.task_uri,
                                        self.source,
                                        "http://example.org/{0}".format(
                                            uuid4()),
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
        """
        Create a language annotation for the source.

        Note: An alternative approach would be to use an "UNK" (unknown) token
        for unsupported languages to preserve metadata that language detection
        was attempted, rather than silently skipping the annotation.
        """
        lang_uri = LANGUAGE_CODE_TO_URI.get(language)
        if not lang_uri:
            error_msg = f"Unsupported language code '{language}' - cannot create language annotation"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        TripletAnnotation(
            subject=source_uri,
            predicate="eli:language",
            obj=sparql_escape_uri(lang_uri),
            activity_id=self.task_uri,
            source_uri=source_uri,
            start=None,
            end=None,
            agent=AI_COMPONENTS["ner_extractor"],
            agent_type=AGENT_TYPES["ai_component"]
        ).add_to_triplestore()

    def create_title_relation(self, source_uri: str, entities: list[dict[str, Any]]):
        for entity in entities:
            if entity['label'] == 'TITLE':
                TripletAnnotation(
                    subject=source_uri,
                    predicate="eli:title",
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

    def create_general_entity_annotations(self, source_uri: str, entities: list[dict[str, Any]]) -> list[str]:
        """
        Create triplet suggestions for entities by mapping (refined) labels to RDF predicates.

        Only entities with a configured mapping are saved. If a label is unmapped
        (or mapped to an empty string), the entity is skipped.
        """
        config = get_config()
        label_to_predicate = config.ner.label_to_predicate or {}
        entity_uris = []

        for entity in entities:
            label = str(entity.get("label", "")).upper()
            predicate = (label_to_predicate.get(label) or "").strip()
            if not predicate:
                continue

            entity_uri = TripletAnnotation(
                subject=source_uri,
                predicate=predicate,
                obj=sparql_escape_string(entity["text"]),
                activity_id=self.task_uri,
                source_uri=source_uri,
                start=entity.get("start"),
                end=entity.get("end"),
                agent=AI_COMPONENTS["ner_extractor"],
                agent_type=AGENT_TYPES["ai_component"],
                confidence=entity.get("confidence", 1.0),
            ).add_to_triplestore()
            entity_uris.append(entity_uri)

        return entity_uris

    def extract_general_entities(self, task_data: str, language: str = None, method: str = None) -> list[dict[str, Any]]:
        """
        Extract general NER entities (PERSON, ORG, DATE, etc.) from text.

        Args:
            task_data: Text to extract entities from
            language: Language for extraction ('nl', 'de', 'en'). Defaults to config.ner.language.
            method: Extraction method ('regex', 'spacy', 'flair', 'composite', 'huggingface'). Defaults to config.ner.method.
        """
        config = get_config()
        if language is None:
            language = config.ner.language
        if method is None:
            method = config.ner.method

        self.logger.info(
            f"Extracting general entities using {method}/{language}")

        # Extract entities using the factory pattern
        entities = extract_entities(
            task_data, language=language, method=method)
        self.logger.info(f"Found {len(entities)} general entities")

        return entities

    def fetch_english_expression_uri(self) -> Optional[str]:
        """
        Find the English expression that realizes the same work as the source expression.
        Returns the URI of the English expression if found, None otherwise.
        """
        work_uri = self.fetch_work_uri()
        if not work_uri:
            self.logger.info(
                f"No work found for expression {self.source}, cannot find English translation")
            return None

        # Get English language URI
        en_lang_uri = LANGUAGE_CODE_TO_URI.get("en")
        if not en_lang_uri:
            self.logger.warning(
                "English language URI not found in LANGUAGE_CODE_TO_URI")
            return None

        # Query for English expression that realizes the same work
        # Check via language annotation (since we use TripletAnnotation for language, not dct:language)
        query_template = Template(
            get_prefixes_for_query("eli", "rdf", "oa") +
            """
            SELECT DISTINCT ?en_expr WHERE {
              # Find expressions that realize the work
              GRAPH ?g {
                ?en_expr a eli:Expression ;
                  eli:realizes $work .
              }
              # And have English language annotation
              GRAPH <""" + GRAPHS["ai"] + """> {
                ?ann oa:hasBody ?stmt .
                ?stmt a rdf:Statement ;
                  rdf:subject ?en_expr ;
                  rdf:predicate eli:language ;
                  rdf:object $en_lang .
              }
            }
            LIMIT 1
            """
        )

        query_result = query(
            query_template.substitute(
                work=sparql_escape_uri(work_uri),
                en_lang=sparql_escape_uri(en_lang_uri)
            ),
            sudo=True
        )

        bindings = query_result.get("results", {}).get("bindings", [])
        if bindings and "en_expr" in bindings[0]:
            en_expr_uri = bindings[0]["en_expr"]["value"]
            self.logger.info(
                f"Found English expression {en_expr_uri} for work {work_uri}")
            return en_expr_uri

        self.logger.info(f"No English expression found for work {work_uri}")
        return None

    def fetch_expression_data(self, expression_uri: str) -> str:
        """
        Retrieve the text content from a specific expression URI.
        Similar to fetch_data() but for a specific expression.
        """
        query_template = Template(
            get_prefixes_for_query("eli", "eli-dl", "dct", "epvoc") +
            """
            SELECT DISTINCT ?title ?description ?decision_basis ?content
            WHERE {
              GRAPH ?graph {
                VALUES ?s {
                  $expression
                }
                OPTIONAL { ?s eli:title ?title }
                OPTIONAL { ?s eli:description ?description }
                OPTIONAL { ?s eli-dl:decision_basis ?decision_basis }
                OPTIONAL { ?s epvoc:expressionContent ?content }
              }
            }
        """)

        query_result = query(
            query_template.substitute(
                expression=sparql_escape_uri(expression_uri)),
            sudo=True
        )

        bindings = query_result.get("results", {}).get("bindings", [])
        texts: list[str] = []
        seen = set()
        for binding in bindings:
            for field in ("content", "title", "description", "decision_basis"):
                value = binding.get(field, {}).get("value")
                if value and value not in seen:
                    texts.append(value)
                    seen.add(value)

        return "\n".join(texts)

    def fetch_eli_expressions(self) -> dict[str, list[str]]:
        """
        Retrieve the ELI expressions and their contents from the task's input container.
        Note that these will be the translated expressions.

        Returns:
            Dictionary containing:
                - "expression_uris": list containing the expression URIs
                - "expression_contents": list containing the expression contents
        """
        q = Template(
            get_prefixes_for_query("task", "epvoc", "eli") +
            f"""
            SELECT ?expression ?content WHERE {{
            GRAPH <{GRAPHS["jobs"]}> {{
                $task task:inputContainer ?container .
            }}

            GRAPH <{GRAPHS["data_containers"]}> {{
                ?container task:hasResource ?expression .
            }}

            GRAPH <{GRAPHS["expressions"]}> {{
                ?expression a eli:Expression ;
                            epvoc:expressionContent ?content .
            }}            
            }}
            """
        ).substitute(task=sparql_escape_uri(self.task_uri))

        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        if not bindings:
            self.logger.warning(
                f"No expressions found in input container for task {self.task_uri}")
            return {
                "expression_uris": [],
                "expression_contents": []
            }

        expression_uris = [b["expression"]["value"] for b in bindings]
        expression_contents = [b["content"]["value"] for b in bindings]

        return {
            "expression_uris": expression_uris,
            "expression_contents": expression_contents
        }

    def create_output_container(self, resource: str) -> str:
        """
        Function to create an output data container for the translated ELI expression.

        Args:
            resource: String containing the URI of the translated ELI expression.

        Returns:
            String containing the URI of the output data container
        """
        container_id = str(uuid.uuid4())
        container_uri = f"http://data.lblod.info/id/data-container/{container_id}"

        q = Template(
            get_prefixes_for_query("task", "nfo", "mu") +
            f"""
            INSERT DATA {{
            GRAPH <{GRAPHS["data_containers"]}> {{
                $container a nfo:DataContainer ;
                    mu:uuid $uuid ;
                    task:hasResource $resource .
            }}
            }}
            """
        ).substitute(
            container=sparql_escape_uri(container_uri),
            uuid=sparql_escape_string(container_id),
            resource=sparql_escape_uri(resource)
        )

        update(q, sudo=True)
        return container_uri

    def process(self):
        """
        Process NER task by retrieving English translation from database and extracting entities.
        Assumes TranslationTask has already run and created the English expression.
        """
        self.logger.info(
            f"Processing entity extraction for source: {self.source}")

        eli_expressions = self.fetch_eli_expressions()

        for i in range(len(eli_expressions["expression_uris"])):
            target_expression_uri = eli_expressions["expression_uris"][i]
            target_english_text = eli_expressions["expression_contents"][i]

            if not target_english_text or not target_english_text.strip():
                self.logger.warning(
                    "No content available for entity extraction")
                return

            # Extract general entities (DATE, etc.) on the English text
            general_entities = self.extract_general_entities(
                target_english_text, language="en")
            entity_uris = self.create_general_entity_annotations(
                target_expression_uri, general_entities)

            for entity_uri in entity_uris:
                self.results_container_uris.append(
                    self.create_output_container(entity_uri))


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

        config = get_config()
        self._llm_config = {
            "model_name": config.llm.model_name,
            "temperature": config.llm.temperature,
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

        if not task_data.strip():
            self.logger.warning(
                "No task data found; skipping model annotation.")
            return

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
                "SDG-15 Life on Land",
                "SDG-16 Peace, Justice and Strong Institutions",
                "SDG-17 Partnerships for the Goals"
                ]

        classes: list[str] = []
        config = get_config()
        api_key = config.llm.api_key.get_secret_value() if config.llm.api_key else None
        if not api_key:
            self.logger.warning(
                "OpenAI API key missing (config.llm.api_key), using dummy SDG label for testing.")
            classes = [random.choice(sdgs).replace(" ", "_")]
        else:
            try:
                llm_input = LlmTaskInput(system_message=self._llm_system_message,
                                         user_message=self._llm_user_message.format(
                                             code_list=sdgs, decision_text=task_data),
                                         assistant_message=None,
                                         output_format=EntityLinkingTaskOutput)

                response = self._llm(llm_input)
                classes = [designated_class.replace(
                    " ", "_") for designated_class in response.designated_classes]
            except Exception as exc:
                self.logger.warning(
                    f"LLM call failed ({exc}); using dummy SDG label for testing.")
                classes = [random.choice(sdgs).replace(" ", "_")]

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
            print(
                f"Processed decision {i+1}/{len(decision_uris)}: {decision_uri}", flush=True)

    def fetch_decisions_without_annotations(self) -> list[str]:
        q = get_prefixes_for_query("rdf", "eli", "oa") + """
        SELECT DISTINCT ?s
        WHERE {
            GRAPH ?dataGraph {
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

        response = query(q, sudo=True)
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

        decisions = [d for d in decisions if d.get("classes")]
        if not decisions:
            print("No labeled decisions found; skipping training.", flush=True)
            return

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
                "SDG-15 Life on Land",
                "SDG-16 Peace, Justice and Strong Institutions",
                "SDG-17 Partnerships for the Goals"
                ]

        config = get_config()
        ml_config = config.ml_training

        print("Started training...", flush=True)
        train(
            decisions[:10],
            sdgs,
            ml_config.huggingface_output_model_id,
            transformer=ml_config.transformer,
            learning_rate=ml_config.learning_rate,
            epochs=ml_config.epochs,
            weight_decay=ml_config.weight_decay,
        )
        print("Done training!", flush=True)

    def convert_classes_to_original_names(self, decisions: list[dict[str, str | list[str]]]):
        for decision in decisions:
            decision["classes"] = [
                c.split("/")[-1].replace("_", " ") for c in decision["classes"]]

        return decisions

    def fetch_decisions_with_classes(self) -> list[dict[str, str | list[str]]]:
        q = get_prefixes_for_query("rdf", "eli", "eli-dl", "oa", "epvoc", "dct") + """
        SELECT ?decision ?title ?description ?decision_basis ?content ?classes
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
            GRAPH ?dataGraph {
                ?decision rdf:type eli:Expression .
                OPTIONAL { ?decision eli:title ?title }
                OPTIONAL { ?decision eli:description ?description }
                OPTIONAL { ?decision eli-dl:decision_basis ?decision_basis }
                OPTIONAL { ?decision epvoc:expressionContent ?content }
                OPTIONAL { ?decision dct:language ?lang }
            }
        }
        """

        res = query(q, sudo=True)
        bindings = res.get("results", {}).get("bindings", [])

        results = []
        for b in bindings:
            decision = b["decision"]["value"]
            classes_concat = b.get("classes", {}).get("value", "")
            classes = [c for c in classes_concat.split("|") if c]
            title = b.get("title", {}).get("value", "")
            description = b.get("description", {}).get("value", "")
            decision_basis = b.get("decision_basis", {}).get("value", "")
            content = b.get("content", {}).get("value", "")

            text = "\n".join(
                [t for t in [title, description, decision_basis, content] if t])

            results.append({
                "decision": decision,
                "classes": classes,
                "text": text
            })

        return results


class TranslationTask(DecisionTask):
    """Task that translates text to a target language using a configurable translation provider."""

    __task_type__ = TASK_OPERATIONS["translation"]

    def __init__(self, task_uri: str):
        super().__init__(task_uri)

        config = get_config()
        self.target_language = config.translation.target_language
        self._translator = None

    def get_translator(self):
        """Lazy load translator based on config.translation.provider."""
        if self._translator is None:
            config = get_config()
            provider = config.translation.provider.lower()
            self.logger.info(f"Initializing translator provider: {provider}")

            if provider == "auto":
                self._translator = Translator()
            else:
                registry = {
                    "google": ("translatepy.translators.google", "GoogleTranslate", True),
                    "microsoft": ("translatepy.translators.microsoft", "MicrosoftTranslate", True),
                    "deepl": ("translatepy.translators.deepl", "DeeplTranslate", True),
                    "libre": ("translatepy.translators.libre", "LibreTranslate", True),
                    "huggingface": ("translation_plugin_huggingface", "HuggingFaceTranslateService", False),
                    "etranslation": ("translation_plugin_etranslation", "ETRanslationService", False),
                    "gemma": ("translation_plugin_gemma", "GemmaTranslateService", False),
                }

                module_name, class_name, is_external = registry.get(
                    provider, registry.get("etranslation", registry["huggingface"]))
                base_package = __package__ or "src"
                module_path = module_name if is_external else f"{base_package}.{module_name}"

                self.logger.info(
                    f"Loading translation module: {module_path}, class: {class_name}, provider: {provider}")
                module = importlib.import_module(module_path)
                service_cls = getattr(module, class_name)
                service = service_cls()
                self._translator = Translator(services_list=[service])

            self.logger.info("Translator initialized successfully")

        return self._translator

    def retrieve_source_language(self, content: Optional[str] = None) -> str:
        """
        Retrieve the source language from the triplestore.
        The language should have been stored by EntityExtractionTask via TripletAnnotation.
        If not found, detect it from the content.

        Args:
            content: Optional content string to use for language detection fallback.
                    If not provided and detection is needed, will fetch data.

        Returns:
            Language code (e.g., 'nl', 'de', 'en')
        """
        query_string = Template(
            get_prefixes_for_query("eli", "rdf", "oa") +
            """
            SELECT ?language WHERE {
                GRAPH <""" + GRAPHS["ai"] + """> {
                    ?ann oa:hasBody ?stmt .
                    ?stmt a rdf:Statement ;
                           rdf:subject <$source> ;
                           rdf:predicate eli:language ;
                           rdf:object ?language .
                }
            }
            LIMIT 1
            """).substitute(source=self.source)

        result = query(query_string, sudo=True)
        bindings = result.get("results", {}).get("bindings", [])
        language_uri = None
        if bindings and "language" in bindings[0] and "value" in bindings[0].get("language", {}):
            language_uri = bindings[0]["language"]["value"]

        if language_uri:
            lang_code = LANGUAGE_URI_TO_CODE.get(language_uri)
            if lang_code:
                self.logger.info(
                    f"Retrieved source language from database: {lang_code}")
                return lang_code
            self.logger.warning(f"Unknown language URI: {language_uri}")

        # Fallback: detect language from content
        self.logger.warning(
            "No language found in database, detecting from content...")
        if not content:
            content = self.fetch_data()

        if not content or not content.strip():
            raise ValueError(
                f"No content available for language detection: {self.source}")

        # Limit text length for language detection to avoid hanging on very long text
        # langdetect can be slow/hang on very long text, so use first 1000 chars
        text_for_detection = content[:1000] if len(content) > 1000 else content
        self.logger.debug(
            f"Using {len(text_for_detection)} chars (of {len(content)}) for language detection")

        try:
            detected_lang = langdetect.detect(text_for_detection)
            self.logger.info(
                f"Detected language from content: {detected_lang}")
            return detected_lang
        except Exception as e:
            self.logger.error(
                f"Language detection failed: {e}. Defaulting to 'nl' (Dutch).")
            return "nl"  # Default to Dutch for Belgian documents

    def create_language_relation(self, source_uri: str, language: str):
        """
        Create a language annotation for the source expression.
        """
        lang_uri = LANGUAGE_CODE_TO_URI.get(language)
        if not lang_uri:
            error_msg = f"Unsupported language code '{language}' - cannot create language annotation"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        TripletAnnotation(
            subject=source_uri,
            predicate="eli:language",
            obj=sparql_escape_uri(lang_uri),
            activity_id=self.task_uri,
            source_uri=source_uri,
            start=None,
            end=None,
            agent=AI_COMPONENTS["translator"],
            agent_type=AGENT_TYPES["ai_component"]
        ).add_to_triplestore()

    def create_translated_expression(self, translated_text: str, target_language: str, work_uri: str) -> str:
        """
        Create an eli:Expression resource for the translated text and
        link it to the same work as the source expression when available.

        Args:
            translated_text: The translated content
            target_language: Target language code (e.g., 'en', 'nl', 'de')
            work_uri: The URI of the work that the translated expression realizes

        Returns:
            URI of the created translated expression
        """
        translated_expr_uri = f"http://example.org/{uuid4()}"
        graph_uri = self.source_graph or GRAPHS["ai"]

        realizes_triple = ""
        if work_uri:
            realizes_triple = f"{sparql_escape_uri(translated_expr_uri)} eli:realizes {sparql_escape_uri(work_uri)} ."

        query_string = (
            get_prefixes_for_query("eli", "epvoc") +
            f"""
            INSERT {{
              GRAPH <{graph_uri}> {{
                {sparql_escape_uri(translated_expr_uri)} a eli:Expression ;
                    epvoc:expressionContent {sparql_escape_string(translated_text)} .
                {realizes_triple}
              }}
            }} WHERE {{
            }}
            """
        )
        update(query_string, sudo=True)

        # Also create an annotated statement that "translated expression realizes work",
        # targeting the original expression via oa:SpecificResource.
        if work_uri:
            TripletAnnotation(
                subject=translated_expr_uri,
                predicate="eli:realizes",
                obj=sparql_escape_uri(work_uri),
                activity_id=self.task_uri,
                source_uri=self.source,
                start=None,
                end=None,
                agent=AI_COMPONENTS["translator"],
                agent_type=AGENT_TYPES["ai_component"],
            ).add_to_triplestore()

        self.logger.info(
            f"Created translated expression {translated_expr_uri} in graph {graph_uri} (language: {target_language})")
        return translated_expr_uri

    def fetch_eli_expressions(self) -> dict[str, list[str]]:
        """
        Retrieve ELI expressions, their epvoc:expressionContent,
        language, and corresponding ELI work URI from the task's input container.

        Returns:
            Dictionary containing:
                - "expression_uris": list containing the expression URIs
                - "expression_contents": list containing the expression contents
                - "languages": list containing the language URIs of the expressions
                - "work_uris": list containing the work URIs of the expressions
        """
        q = Template(
            get_prefixes_for_query("task", "epvoc", "eli") +
            f"""
            SELECT ?expression ?content ?lang ?work WHERE {{
            GRAPH <{GRAPHS["jobs"]}> {{
                $task task:inputContainer ?container .
            }}

            GRAPH <{GRAPHS["data_containers"]}> {{
                ?container task:hasResource ?expression .
            }}

            GRAPH <{GRAPHS["expressions"]}> {{
                ?expression a eli:Expression ;
                            epvoc:expressionContent ?content ;
                            eli:language ?lang .
            }}

            GRAPH <{GRAPHS["works"]}> {{
                ?work a eli:Work ;
                    eli:is_realized_by ?expression .
            }}
            }}
            """
        ).substitute(task=sparql_escape_uri(self.task_uri))

        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        if not bindings:
            self.logger.warning(
                f"No expressions found in input container for task {self.task_uri}")
            return {
                "expression_uris": [],
                "expression_contents": [],
                "languages": [],
                "work_uris": [],
            }

        expression_uris = [b["expression"]["value"] for b in bindings]
        expression_contents = [b["content"]["value"] for b in bindings]
        languages = [b["lang"]["value"] for b in bindings]
        work_uris = [b["work"]["value"] for b in bindings]

        return {
            "expression_uris": expression_uris,
            "expression_contents": expression_contents,
            "languages": languages,
            "work_uris": work_uris,
        }

    def create_output_container(self, resource: str) -> str:
        """
        Function to create an output data container for the translated ELI expression.

        Args:
            resource: String containing the URI of the translated ELI expression.

        Returns:
            String containing the URI of the output data container
        """
        container_id = str(uuid.uuid4())
        container_uri = f"http://data.lblod.info/id/data-container/{container_id}"

        q = Template(
            get_prefixes_for_query("task", "nfo", "mu") +
            f"""
            INSERT DATA {{
            GRAPH <{GRAPHS["data_containers"]}> {{
                $container a nfo:DataContainer ;
                    mu:uuid $uuid ;
                    task:hasResource $resource .
            }}
            }}
            """
        ).substitute(
            container=sparql_escape_uri(container_uri),
            uuid=sparql_escape_string(container_id),
            resource=sparql_escape_uri(resource)
        )

        update(q, sudo=True)
        return container_uri

    def process(self):
        """
        Main processing logic: fetch text, detect language, annotate source expression,
        translate it, and store translated expression in triplestore.
        """
        # Fetch the original text
        eli_expressions = self.fetch_eli_expressions()

        for i in range(len(eli_expressions["expression_uris"])):
            original_text = eli_expressions["expression_contents"][i]
            source_language = LANGUAGE_URI_TO_CODE.get(
                eli_expressions["languages"][i], None)
            work_uri = eli_expressions["work_uris"][i]
            self.logger.info(
                f"Processing translation for source: {self.source}")

            # If there's no content, skip early
            if not original_text or not original_text.strip():
                self.logger.warning(
                    "No content found for source; skipping translation")
                return

            # Ensure we have an explicit source language
            if not source_language or source_language.lower() == "auto":
                self.logger.warning(
                    "Source language was 'auto' or missing, detecting from content...")
                try:
                    # Limit text length for language detection to avoid hanging on very long text
                    text_for_detection = original_text[:1000] if len(
                        original_text) > 1000 else original_text
                    self.logger.debug(
                        f"Using {len(text_for_detection)} chars (of {len(original_text)}) for language detection")
                    source_language = langdetect.detect(text_for_detection)
                    self.logger.info(
                        f"Detected source language: {source_language}")
                except Exception as e:
                    self.logger.error(
                        f"Language detection failed: {e}. Defaulting to 'nl' (Dutch).")
                    source_language = "nl"  # Default to Dutch for Belgian documents

            # Skip translation if already in target language
            if source_language.lower() == self.target_language.lower():
                self.logger.info(
                    f"Text is already in target language ({self.target_language}), skipping translation")
                return

            # Get translator and translate
            translator = self.get_translator()

            self.logger.info(
                f"Translating from {source_language} to {self.target_language}")

            # Use translatepy's translate method
            translation_result = translator.translate(
                original_text,
                destination_language=self.target_language,
                source_language=source_language
            )

            # Extract the translated text from the result
            translated_text = translation_result.result
            service_used = translation_result.service
            self.logger.info(f"Translation service used: {service_used}")

            self.logger.info(f"Translation completed.")

            # Create a new eli:Expression for the translated text
            normalized_target_lang = self.target_language.lower()
            translated_expression_uri = self.create_translated_expression(
                translated_text, normalized_target_lang, work_uri)

            # Annotate the translated expression with its language
            self.create_language_relation(
                translated_expression_uri, normalized_target_lang)

            self.logger.info(
                f"Translation stored as new expression: {translated_expression_uri}")

            self.results_container_uris.append(
                self.create_output_container(translated_expression_uri))


# SegmentationTask and its variants both follow the same pattern: fetch text, run segmentor, store spans as annotations.
class SegmentationTask(DecisionTask):
    """Run the marked segmentation model and store segment spans as annotations."""

    __task_type__ = TASK_OPERATIONS["segmentation"]

    def fetch_english_expression_uri(self) -> Optional[str]:
        """
        Find the English expression that realizes the same work as the source expression.
        Returns the URI of the English expression if found, None otherwise.
        """
        work_uri = self.fetch_work_uri()
        if not work_uri:
            self.logger.info(
                f"No work found for expression {self.source}, cannot find English translation")
            return None

        # Get English language URI
        en_lang_uri = LANGUAGE_CODE_TO_URI.get("en")
        if not en_lang_uri:
            self.logger.warning(
                "English language URI not found in LANGUAGE_CODE_TO_URI")
            return None

        # Query for English expression that realizes the same work
        # Check via language annotation (since we use TripletAnnotation for language, not dct:language)
        query_template = Template(
            get_prefixes_for_query("eli", "rdf", "oa") +
            """
            SELECT DISTINCT ?en_expr WHERE {
              # Find expressions that realize the work
              GRAPH ?g {
                ?en_expr a eli:Expression ;
                  eli:realizes $work .
              }
              # And have English language annotation
              GRAPH <""" + GRAPHS["ai"] + """> {
                ?ann oa:hasBody ?stmt .
                ?stmt a rdf:Statement ;
                  rdf:subject ?en_expr ;
                  rdf:predicate eli:language ;
                  rdf:object $en_lang .
              }
            }
            LIMIT 1
            """
        )

        query_result = query(
            query_template.substitute(
                work=sparql_escape_uri(work_uri),
                en_lang=sparql_escape_uri(en_lang_uri)
            ),
            sudo=True
        )

        bindings = query_result.get("results", {}).get("bindings", [])
        if bindings and "en_expr" in bindings[0]:
            en_expr_uri = bindings[0]["en_expr"]["value"]
            self.logger.info(
                f"Found English expression {en_expr_uri} for work {work_uri}")
            return en_expr_uri

        self.logger.info(f"No English expression found for work {work_uri}")
        return None

    def fetch_expression_data(self, expression_uri: str) -> str:
        """
        Retrieve the text content from a specific expression URI.
        Similar to fetch_data() but for a specific expression.
        """
        query_template = Template(
            get_prefixes_for_query("eli", "eli-dl", "dct", "epvoc") +
            """
            SELECT DISTINCT ?title ?description ?decision_basis ?content
            WHERE {
              GRAPH ?graph {
                VALUES ?s {
                  $expression
                }
                OPTIONAL { ?s eli:title ?title }
                OPTIONAL { ?s eli:description ?description }
                OPTIONAL { ?s eli-dl:decision_basis ?decision_basis }
                OPTIONAL { ?s epvoc:expressionContent ?content }
              }
            }
        """)

        query_result = query(
            query_template.substitute(
                expression=sparql_escape_uri(expression_uri)),
            sudo=True
        )

        bindings = query_result.get("results", {}).get("bindings", [])
        texts: list[str] = []
        seen = set()
        for binding in bindings:
            for field in ("content", "title", "description", "decision_basis"):
                value = binding.get(field, {}).get("value")
                if value and value not in seen:
                    texts.append(value)
                    seen.add(value)

        return "\n".join(texts)

    def _create_segmentor(self):
        """Create a Segmentor configured from app config."""
        from .library.segmentors import GemmaSegmentor, LLMSegmentor
        seg_config = get_config().segmentation
        api_key = seg_config.api_key.get_secret_value() if seg_config.api_key else None

        if seg_config.model_name == "wdmuer/decide-marked-segmentation":
            return GemmaSegmentor(
                api_key=api_key,
                endpoint=seg_config.endpoint,
                model_name=seg_config.model_name,
                temperature=seg_config.temperature,
                max_new_tokens=seg_config.max_new_tokens,
            )
        else:
            return LLMSegmentor(
                api_key=api_key,
                endpoint=seg_config.endpoint,
                model_name=seg_config.model_name,
                temperature=seg_config.temperature,
                max_new_tokens=seg_config.max_new_tokens,
            )

    def create_segment_annotations(self, source_uri: str, segments: list[dict[str, Any]]):
        """
        Store segment spans as TripletAnnotation.
        Creates rdf:Statement with subject=expression, predicate=segment_type, object=segment_text.
        """
        for segment in segments:
            segment_label = segment.get('label', 'UNKNOWN')
            segment_text = segment.get('text', '')

            # Use segment label as predicate (e.g., "ex:TITLE", "ex:PARTICIPANTS")
            # Convert to proper predicate format
            predicate = f"ex:{segment_label}"

            TripletAnnotation(
                subject=source_uri,
                predicate=predicate,
                obj=sparql_escape_string(segment_text),
                activity_id=self.task_uri,
                source_uri=source_uri,
                start=segment.get('start'),
                end=segment.get('end'),
                agent=AI_COMPONENTS["segmenter"],
                agent_type=AGENT_TYPES["ai_component"],
                confidence=1.0
            ).add_to_triplestore()
            self.logger.info(
                f"Created segment annotation for '{segment_label}' at [{segment.get('start')}:{segment.get('end')}] with text: '{segment_text[:50]}...'")

    def process(self):
        """
        Fetch English text through original expression (if translation available),
        run the segmentor, and save annotations on the English expression.
        """
        self.logger.info(f"Processing segmentation for {self.source}")

        # Try to find English expression that realizes the same work
        english_expression_uri = self.fetch_english_expression_uri()

        if english_expression_uri:
            # Use English expression for segmentation
            self.logger.info(
                f"Found English expression {english_expression_uri}, using it for segmentation")
            task_data = self.fetch_expression_data(english_expression_uri)
            target_expression_uri = english_expression_uri
        else:
            # Fallback to original expression if no English translation available
            self.logger.info(
                f"No English expression found, using original expression {self.source}")
            task_data = self.fetch_data()
            target_expression_uri = self.source

        if not task_data or not task_data.strip():
            self.logger.warning("No content available for segmentation")
            return

        segmentor = self._create_segmentor()
        segments = segmentor.segment(task_data)
        self.logger.info(f"Segmentor returned {len(segments)} segments")

        # Save annotations on the target expression (English if available, otherwise original)
        self.create_segment_annotations(target_expression_uri, segments)
