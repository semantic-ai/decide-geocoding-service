import contextlib
import logging
import os
import importlib
import json
import langdetect
from abc import ABC, abstractmethod
from typing import Optional, Any

from uuid import uuid4
from helpers import query
from string import Template
from translatepy import Translator
from escape_helpers import sparql_escape_uri, sparql_escape_string

from .helper_functions import clean_string, get_start_end_offsets, process_text, geocode_detectable
from .ner_extractors import SpacyGeoAnalyzer
from .ner_functions import extract_entities
from .nominatim_geocoder import NominatimGeocoder
from .annotation import GeoAnnotation, LinkingAnnotation, TripletAnnotation
from .sparql_config import get_prefixes_for_query, GRAPHS, JOB_STATUSES, TASK_OPERATIONS, AI_COMPONENTS, AGENT_TYPES, LANGUAGE_CODE_TO_URI, LANGUAGE_URI_TO_CODE
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

        # Handle optional fields - they may not exist in the result
        binding = query_result["results"]["bindings"][0] if query_result["results"]["bindings"] else {}
        
        title = binding.get("title", {}).get("value", "") if binding.get("title") else ""
        description = binding.get("description", {}).get("value", "") if binding.get("description") else ""
        decision_basis = binding.get("decision_basis", {}).get("value", "") if binding.get("decision_basis") else ""

        # Join non-empty parts
        parts = [part for part in [title, description, decision_basis] if part]
        return "\n".join(parts) if parts else ""


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
        TripletAnnotation(
            subject=self.source,
            predicate="eli:language",
            obj=sparql_escape_uri(LANGUAGE_CODE_TO_URI.get(language)),
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

        # Uses defaults from ner_config.py: language='dutch', method='regex'
        # Language can be passed in future when extracted from database
        #entities = self.extract_general_entities(eli_expression, language)

        # todo to be improved upon a lot by inserting functions to create the other predicates
        # ELI properties
        entities = extract_entities(eli_expression, language=language, method='title')
        self.create_title_relation(self.source, entities)


class ModelAnnotatingTask(DecisionTask):
    """Task that links the correct code from a list to text."""
    
    __task_type__ = TASK_OPERATIONS["model_annotation"]

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


class TranslationTask(DecisionTask):
    """Task that translates text to a target language using a configurable translation provider."""
    
    __task_type__ = TASK_OPERATIONS["translation"]
    
    def __init__(self, task_uri: str):
        super().__init__(task_uri)
        
        self.target_language = os.getenv("TRANSLATION_TARGET_LANG", "en")
        self._translator = None
    
    def get_translator(self):
        """Lazy load translator based on TRANSLATION_PROVIDER env var."""
        if self._translator is None:
            provider = os.getenv("TRANSLATION_PROVIDER", "huggingface").lower()
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
        language_uri = bindings[0]["language"]["value"] if bindings else None
        
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
        TripletAnnotation(
            subject=translation_annotation_uri,
            predicate="eli:language",
            obj=sparql_escape_uri(LANGUAGE_CODE_TO_URI[target_language]),
            activity_id=self.task_uri,
            source_uri=translation_annotation_uri,
            start=0,
            end=0,
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
            start=0,
            end=0,
            agent=AI_COMPONENTS["translator"],
            agent_type=AGENT_TYPES["ai_component"]
        ).add_to_triplestore()
        
        # Normalize language code (lowercase) and verify it exists in our mapping
        normalized_lang = target_language.lower()
        if normalized_lang in LANGUAGE_CODE_TO_URI:
            # Create language annotation that targets the translation annotation itself
            self.create_target_language_relation(translation_annotation_uri, normalized_lang)
        else:
            self.logger.warning(f"Language code '{target_language}' not found in LANGUAGE_CODE_TO_URI mapping, skipping language annotation")
        
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
