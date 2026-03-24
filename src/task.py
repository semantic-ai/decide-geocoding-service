import importlib
import uuid
import langdetect
from typing import Optional, Any

from uuid import uuid4
from helpers import query, update
from string import Template
from translatepy import Translator
from escape_helpers import sparql_escape_uri, sparql_escape_string

from decide_ai_service_base.task import Task
from decide_ai_service_base.sparql_config import get_prefixes_for_query, GRAPHS, TASK_OPERATIONS, AI_COMPONENTS, AGENT_TYPES, LANGUAGE_CODE_TO_URI, LANGUAGE_URI_TO_CODE
from decide_ai_service_base.annotation import RelationExtractionAnnotation

from .ner_functions import extract_entities
from .config import get_config

# Projection utilities for Segments and Entity alignment
from .library.entity_projections import project_spans


class EntityExtractionTask(Task):
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
        RelationExtractionAnnotation(
            subject=source_uri,
            predicate="eli:language",
            obj=sparql_escape_uri(lang_uri),
            activity_id=self.task_uri,
            source_uri=source_uri,
            start=None,
            end=None,
            agent=AI_COMPONENTS["ner_extractor"],
            agent_type=AGENT_TYPES["ai_component"]
        ).add_to_triplestore_if_not_exists()

    def create_title_relation(self, source_uri: str, entities: list[dict[str, Any]]):
        print(f"[TITLE] subject: {self.source}; source_uri: {source_uri}")

        for entity in entities:
            if entity['label'] == 'TITLE':
                RelationExtractionAnnotation(
                    subject=source_uri,
                    predicate="eli:title",
                    obj=sparql_escape_string(entity['text']),
                    activity_id=self.task_uri,
                    source_uri=source_uri,
                    start=entity['start'],
                    end=entity['end'],
                    agent=AI_COMPONENTS["ner_extractor"],
                    agent_type=AGENT_TYPES["ai_component"]
                ).add_to_triplestore_if_not_exists()
                self.logger.info(
                    f"Created Title triplet suggestion for '{entity['text']}' ({entity['label']}) at [{entity['start']}:{entity['end']}]")

    def create_general_entity_annotations(self, source_uri: str, entities: list[dict[str, Any]]) -> list[str]:
        """
        Create triplet suggestions for entities by mapping (refined) labels to RDF predicates.

        Only entities with a configured mapping are saved. If a label is unmapped
        (or mapped to an empty string), the entity is skipped.
        """
        print(f"[GENERAL ENTITIES] subject: {self.source}; source_uri: {source_uri}")

        config = get_config()
        label_to_predicate = config.ner.label_to_predicate or {}
        entity_uris = []

        for entity in entities:
            label = str(entity.get("label", "")).upper()
            predicate = (label_to_predicate.get(label) or "").strip()
            if not predicate:
                continue

            entity_uri = RelationExtractionAnnotation(
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
                entity_class=label
            ).add_to_triplestore_if_not_exists()
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
              GRAPH $graph {
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
                en_lang=sparql_escape_uri(en_lang_uri),
                graph=sparql_escape_uri(GRAPHS["ai"])
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
        ).substitute(
            task=sparql_escape_uri(self.task_uri)
        )

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
        eli_expressions = self.fetch_eli_expressions()

        for i in range(len(eli_expressions["expression_uris"])):
            target_expression_uri = eli_expressions["expression_uris"][i]
            target_english_text = eli_expressions["expression_contents"][i]

            self.logger.info(
                f"Processing entity extraction for source: {target_expression_uri}")

            if not target_english_text or not target_english_text.strip():
                self.logger.warning(
                    "No content available for entity extraction")
                continue

            # Extract general entities (DATE, etc.) on the English text
            general_entities = self.extract_general_entities(
                target_english_text, language="en")
            entity_uris = self.create_general_entity_annotations(
                target_expression_uri, general_entities)

            for entity_uri in entity_uris:
                self.results_container_uris.append(
                    self.create_output_container(entity_uri))

            # Projection back to original/source expression
            if not general_entities:
                continue

            source_expression_uri, source_text = self.resolve_projection_context(target_expression_uri,translated_text=target_english_text)
            if not source_text or source_expression_uri == target_expression_uri:
                continue

            general_entities_projected = project_spans(target_english_text,source_text,general_entities)
            entity_uris_projected = self.create_general_entity_annotations(source_expression_uri, general_entities_projected)
            for projected_entity_uri in entity_uris_projected:
                self.results_container_uris.append(
                    self.create_output_container(projected_entity_uri))


class TranslationTask(Task):
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
        RelationExtractionAnnotation(
            subject=source_uri,
            predicate="eli:language",
            obj=sparql_escape_uri(lang_uri),
            activity_id=self.task_uri,
            source_uri=source_uri,
            start=None,
            end=None,
            agent=AI_COMPONENTS["translator"],
            agent_type=AGENT_TYPES["ai_component"]
        ).add_to_triplestore_if_not_exists()

    def create_translated_expression(
        self,
        translated_text: str,
        target_language: str,
        work_uri: str,
        source_expression_uri: str,
        graph_uri: Optional[str] = None
    ) -> str:
        """
        Create an eli:Expression resource for the translated text and
        link it to the same work as the source expression when available.

        Args:
            translated_text: The translated content
            target_language: Target language code (e.g., 'en', 'nl', 'de')
            work_uri: The URI of the work that the translated expression realizes
            source_expression_uri: The original expression URI used as provenance source
            graph_uri: Optional graph where translated expression is stored

        Returns:
            URI of the created translated expression
        """
        translated_expr_uri = f"http://example.org/{uuid4()}"
        target_graph_uri = graph_uri or GRAPHS["ai"]

        realizes_triple = ""
        if work_uri:
            realizes_triple = f"{sparql_escape_uri(translated_expr_uri)} eli:realizes {sparql_escape_uri(work_uri)} ."

        query_string = (
            get_prefixes_for_query("eli", "epvoc") +
            f"""
            INSERT {{
              GRAPH <{target_graph_uri}> {{
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
            RelationExtractionAnnotation(
                subject=translated_expr_uri,
                predicate="eli:realizes",
                obj=sparql_escape_uri(work_uri),
                activity_id=self.task_uri,
                source_uri=source_expression_uri,
                start=None,
                end=None,
                agent=AI_COMPONENTS["translator"],
                agent_type=AGENT_TYPES["ai_component"],
            ).add_to_triplestore_if_not_exists()

        self.logger.info(
            f"Created translated expression {translated_expr_uri} in graph {target_graph_uri} (language: {target_language})")
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
            source_expression_uri = eli_expressions["expression_uris"][i]
            original_text = eli_expressions["expression_contents"][i]
            source_language = LANGUAGE_URI_TO_CODE.get(
                eli_expressions["languages"][i], None)
            work_uri = eli_expressions["work_uris"][i]
            self.logger.info(
                f"Processing translation for source: {source_expression_uri}")

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
                translated_text,
                normalized_target_lang,
                work_uri,
                source_expression_uri
            )

            # Annotate the translated expression with its language
            self.create_language_relation(
                translated_expression_uri, normalized_target_lang)

            self.logger.info(
                f"Translation stored as new expression: {translated_expression_uri}")

            self.results_container_uris.append(
                self.create_output_container(translated_expression_uri))

# SegmentationTask and its variants both follow the same pattern: fetch text, run segmentor, store spans as annotations.
class SegmentationTask(Task):
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

    def create_segment_annotations(self, source_uri: str, segments: list[dict[str, Any]]) -> list[str]:
        """
        Store segment spans as TripletAnnotation.
        Creates rdf:Statement with subject=expression, predicate=segment_type, object=segment_text.

        Args:
            source_uri: URI of the expression being segmented
            segments: List of segment dicts, each containing 'label', 'text', 'start', 'end'

        Returns:
            List of URIs of the created segment annotations
        """
        segment_uris = []

        for segment in segments:
            segment_label = segment.get(
                'label', 'UNKNOWN').replace(" ", "_").lower()
            segment_text = segment.get('text', '')

            # Use segment label as predicate (e.g., "ex:TITLE", "ex:PARTICIPANTS")
            # Convert to proper predicate format

            predicate = f"ex:{segment_label}"
            segment_uri = RelationExtractionAnnotation(
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
            ).add_to_triplestore_if_not_exists()
            segment_uris.append(segment_uri)

            self.logger.info(
                f"Created segment annotation for '{segment_label}' at [{segment.get('start')}:{segment.get('end')}] with text: '{segment_text[:50]}...'")

        return segment_uris

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
        Fetch English text through original expression (if translation available),
        run the segmentor, and save annotations on the English expression.
        """
        eli_expressions = self.fetch_eli_expressions()

        for i in range(len(eli_expressions["expression_uris"])):
            target_expression_uri = eli_expressions["expression_uris"][i]
            target_english_text = eli_expressions["expression_contents"][i]

            self.logger.info(
                f"Processing segmentation for {target_expression_uri}")

            if not target_english_text or not target_english_text.strip():
                self.logger.warning("No content available for segmentation")
                continue

            # Segment English text
            segmentor = self._create_segmentor()
            segments = segmentor.segment(target_english_text)
            self.logger.info(f"Segmentor returned {len(segments)} segments")

            segment_uris = self.create_segment_annotations(target_expression_uri, segments)
            for segment_uri in segment_uris:
                self.results_container_uris.append(self.create_output_container(segment_uri))

            # pass expression forward to NER task
            self.results_container_uris.append(
                self.create_output_container(target_expression_uri)
            )

            # Project segments back to original/source expression
            if not segments:
                continue

            source_expression_uri, source_text = self.resolve_projection_context(
                target_expression_uri,
                translated_text=target_english_text,
            )
            if source_expression_uri == target_expression_uri:
                continue

            projected_segments = project_spans(
                target_english_text,
                source_text,
                segments,
                max_gap=get_config().segmentation.max_gap,
            )

            projected_segment_uris = self.create_segment_annotations(
                source_expression_uri, projected_segments
            )
            for projected_segment_uri in projected_segment_uris:
                self.results_container_uris.append(
                    self.create_output_container(projected_segment_uri)
                )