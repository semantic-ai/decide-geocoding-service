import contextlib
import logging
import os
import importlib
import json
import langdetect
import random
from abc import ABC, abstractmethod
from typing import Optional, Any, Type

from uuid import uuid4
from helpers import query
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
        try:
            yield
            self.change_state("busy", "success")
        except Exception as e:
            self.logger.error(f"Task {self.task_uri} failed: {type(e).__name__}: {str(e)}", exc_info=True)
            try:
                self.change_state("busy", "failed")
            except Exception as state_error:
                self.logger.error(f"Failed to update task {self.task_uri} status to failed: {state_error}")
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
        bindings = r.get("results", {}).get("bindings", [])
        if not bindings or "source" not in bindings[0] or "value" not in bindings[0].get("source", {}):
            raise ValueError(f"No source found for task {task_uri}")
        self.source = bindings[0]["source"]["value"]

    def fetch_data(self) -> str:
        """Retrieve the input data for this task from the triplestore."""
        query_template = Template(
            get_prefixes_for_query("eli", "eli-dl", "dct", "epvoc") +
            """
            SELECT ?graph ?title ?description ?decision_basis ?content ?lang
            WHERE {
              GRAPH ?graph {
                BIND($source AS ?s)
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
        ))

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


class GeoExtractionTask(DecisionTask):
    """Task that geocodes location information from text."""

    __task_type__ = TASK_OPERATIONS["geo_extraction"]
    config = get_config()
    ner_analyzer = SpacyGeoAnalyzer(model_path=os.getenv("NER_MODEL_PATH"), labels=config.ner.labels)
    geocoder = NominatimGeocoder(base_url=str(config.geocoding.nominatim_base_url), rate_limit=0.5)

    def apply_geo_entities(self, task_data: str):
        """Extract geographic entities from text and store as annotations."""
        default_city = "Gent"

        cleaned_text = clean_string(task_data)
        detectables, _, doc = process_text(cleaned_text, self.__class__.ner_analyzer, default_city)

        if hasattr(doc, 'error'):
            self.logger.error(f"Error: {doc['error']}")
        else:
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
            subject=self.source,
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

    def create_general_entity_annotations(self, source_uri: str, entities: list[dict[str, Any]]):
        """
        Create triplet suggestions for entities by mapping (refined) labels to RDF predicates.
        
        Only entities with a configured mapping are saved. If a label is unmapped
        (or mapped to an empty string), the entity is skipped.
        """
        config = get_config()
        label_to_predicate = config.ner.label_to_predicate or {}

        for entity in entities:
            label = str(entity.get("label", "")).upper()
            predicate = (label_to_predicate.get(label) or "").strip()
            if not predicate:
                continue

            TripletAnnotation(
                subject=self.source,
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
        
        self.logger.info(f"Extracting general entities using {method}/{language}")
        
        # Extract entities using the factory pattern
        entities = extract_entities(task_data, language=language, method=method)
        self.logger.info(f"Found {len(entities)} general entities")

        return entities

    def process(self):
        eli_expression = self.fetch_data()
        self.logger.info(eli_expression)

        # whitespace-only content can crash langdetect
        if not eli_expression or not eli_expression.strip():
            self.logger.warning("No content available for language detection; skipping language annotation")
            return

        try:
            language = langdetect.detect(eli_expression)
        except Exception as e:
            self.logger.warning(f"Language detection failed ({e}); defaulting to 'en'")
            language = "en"

        self.create_language_relation(self.source, language)

        # Extract title using LLM-based title extraction
        title_entities = extract_entities(eli_expression, language=language, method='title')
        self.create_title_relation(self.source, title_entities)
        
        # Extract general entities (DATE, etc.) - uses config.ner.method from config.json
        general_entities = self.extract_general_entities(eli_expression, language=language)
        self.create_general_entity_annotations(self.source, general_entities)


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
            self.logger.warning("No task data found; skipping model annotation.")
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
            self.logger.warning("OpenAI API key missing (config.llm.api_key), using dummy SDG label for testing.")
            classes = [random.choice(sdgs).replace(" ", "_")]
        else:
            try:
                llm_input = LlmTaskInput(system_message=self._llm_system_message,
                                         user_message=self._llm_user_message.format(
                                             code_list=sdgs, decision_text=task_data),
                                         assistant_message=None,
                                         output_format=EntityLinkingTaskOutput)

                response = self._llm(llm_input)
                classes = [designated_class.replace(" ", "_") for designated_class in response.designated_classes]
            except Exception as exc:
                self.logger.warning(f"LLM call failed ({exc}); using dummy SDG label for testing.")
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
            print(f"Processed decision {i+1}/{len(decision_uris)}: {decision_uri}", flush=True)

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
            decision["classes"] = [c.split("/")[-1].replace("_", " ") for c in decision["classes"]]

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
            content = b.get("content", {}).get("value", "")

            text = "\n".join([t for t in [title, description, decision_basis, content] if t])

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

                module_name, class_name, is_external = registry.get(provider, registry.get("etranslation", registry["huggingface"])) 
                base_package = __package__ or "src"
                module_path = module_name if is_external else f"{base_package}.{module_name}"

                self.logger.info(f"Loading translation module: {module_path}, class: {class_name}, provider: {provider}")
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
        
        result = query(query_string)
        bindings = result.get("results", {}).get("bindings", [])
        language_uri = None
        if bindings and "language" in bindings[0] and "value" in bindings[0].get("language", {}):
            language_uri = bindings[0]["language"]["value"]
        
        if language_uri:
            lang_code = LANGUAGE_URI_TO_CODE.get(language_uri)
            if lang_code:
                self.logger.info(f"Retrieved source language from database: {lang_code}")
                return lang_code
            self.logger.warning(f"Unknown language URI: {language_uri}")
        
        # Fallback: detect language from content
        self.logger.warning("No language found in database, detecting from content...")
        if not content:
            content = self.fetch_data()
        
        if not content or not content.strip():
            raise ValueError(f"No content available for language detection: {self.source}")
        
        # Limit text length for language detection to avoid hanging on very long text
        # langdetect can be slow/hang on very long text, so use first 1000 chars
        text_for_detection = content[:1000] if len(content) > 1000 else content
        self.logger.debug(f"Using {len(text_for_detection)} chars (of {len(content)}) for language detection")
        
        try:
            detected_lang = langdetect.detect(text_for_detection)
            self.logger.info(f"Detected language from content: {detected_lang}")
            return detected_lang
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}. Defaulting to 'nl' (Dutch).")
            return "nl"  # Default to Dutch for Belgian documents
    
    def create_target_language_relation(self, translation_annotation_uri: str, target_language: str):
        """
        Store the target language as an annotation of the translation annotation itself.
        Uses eli:language predicate (same as EntityExtractionTask uses for source language).
        
        Args:
            translation_annotation_uri: URI of the translation annotation to annotate
            target_language: Target language code (must exist in LANGUAGE_CODE_TO_URI)
        """
        lang_uri = LANGUAGE_CODE_TO_URI.get(target_language)
        if not lang_uri:
            error_msg = f"Unsupported target language '{target_language}' - cannot create language annotation"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        TripletAnnotation(
            subject=translation_annotation_uri,
            predicate="eli:language",
            obj=sparql_escape_uri(lang_uri),
            activity_id=self.task_uri,
            source_uri=translation_annotation_uri,
            start=None,
            end=None,
            agent=AI_COMPONENTS["translator"],
            agent_type=AGENT_TYPES["ai_component"]
        ).add_to_triplestore()
        self.logger.info(f"Stored target language annotation on translation annotation: {target_language}")
    
    def create_translation_relation(self, source_uri: str, translated_text: str, target_language: str):
        """
        Store the translated text as an annotation of the original expression.
        Then create a language annotation that targets the translation annotation itself.
        
        Args:
            source_uri: URI of the source expression being annotated
            translated_text: The translated content (stored as string literal, like title)
            target_language: Target language code
        """
        # Store translation text directly as annotation (same pattern as title)
        # The text is stored as a string literal in rdf:object, just like title does
        translation_annotation_uri = TripletAnnotation(
            subject=self.source,
            predicate="ex:hasTranslation",
            obj=sparql_escape_string(translated_text),
            activity_id=self.task_uri,
            source_uri=source_uri,
            start=None,
            end=None,
            agent=AI_COMPONENTS["translator"],
            agent_type=AGENT_TYPES["ai_component"]
        ).add_to_triplestore()
        
        # Normalize language code (lowercase) and verify it exists in our mapping
        normalized_lang = target_language.lower()
        if normalized_lang not in LANGUAGE_CODE_TO_URI:
            error_msg = f"Unsupported target language '{target_language}' - cannot create language annotation"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        # Create language annotation that targets the translation annotation itself
        self.create_target_language_relation(translation_annotation_uri, normalized_lang)
        
        self.logger.info(f"Stored translation text as annotation (language: {target_language})")
    
    def process(self):
        """
        Main processing logic: fetch text, translate it, store in triplestore.
        """
        # Fetch the original text
        original_text = self.fetch_data()
        self.logger.info(f"Processing translation for source: {self.source}")

        # If there's no content, skip early
        if not original_text or not original_text.strip():
            self.logger.warning("No content found for source; skipping translation")
            return
        
        # Retrieve source language from database (set by NER task)
        source_language = self.retrieve_source_language(content=original_text)
        
        # Ensure we have an explicit source language
        if not source_language or source_language.lower() == "auto":
            self.logger.warning("Source language was 'auto' or missing, detecting from content...")
            try:
                # Limit text length for language detection to avoid hanging on very long text
                text_for_detection = original_text[:1000] if len(original_text) > 1000 else original_text
                self.logger.debug(f"Using {len(text_for_detection)} chars (of {len(original_text)}) for language detection")
                source_language = langdetect.detect(text_for_detection)
                self.logger.info(f"Detected source language: {source_language}")
            except Exception as e:
                self.logger.error(f"Language detection failed: {e}. Defaulting to 'nl' (Dutch).")
                source_language = "nl"  # Default to Dutch for Belgian documents
        
        # Skip translation if already in target language
        if source_language.lower() == self.target_language.lower():
            self.logger.info(f"Text is already in target language ({self.target_language}), skipping translation")
            return
        
        # Get translator and translate
        translator = self.get_translator()
        
        self.logger.info(f"Translating from {source_language} to {self.target_language}")
        
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
        
        # Store the translation as annotation (same pattern as title/language - text stored as string literal)
        self.create_translation_relation(
            self.source,
            translated_text,
            self.target_language
        )
        
        self.logger.info(f"Translation stored as annotation")