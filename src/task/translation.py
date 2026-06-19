import importlib
import uuid
import langdetect
from typing import Optional

from uuid import uuid4
from helpers import query, update, logger
from string import Template
from translatepy import Translator
from escape_helpers import sparql_escape_uri, sparql_escape_string

from decide_ai_service_base.task import DecisionTask
from decide_ai_service_base.sparql_config import get_prefixes_for_query, SPARQL_PREFIXES, GRAPHS, TASK_OPERATIONS, AGENT_TYPES, LANGUAGE_CODE_TO_URI, LANGUAGE_URI_TO_CODE
from decide_ai_service_base.util import (
    get_agent_uri
)
from decide_ai_service_base.annotation import RelationExtractionAnnotation

from ..config import get_config


class TranslationTask(DecisionTask):
    """Task that translates text to a target language using a configurable translation provider."""

    __task_type__ = TASK_OPERATIONS["translation"]

    def __init__(self, task_uri: str, *args, **kwargs):
        super().__init__(task_uri, *args, **kwargs)

        config = get_config()
        self.target_language = config.translation.target_language
        self._translator = None

    def get_translator(self):
        """Lazy load translator based on config.translation.provider."""
        if self._translator is None:
            config = get_config()
            provider = config.translation.provider.lower()
            logger.info(f"Initializing translator provider: {provider}")

            registry = {
                "huggingface": ("translation_plugin_huggingface", "HuggingFaceTranslateService"),
                "etranslation": ("translation_plugin_etranslation", "ETRanslationService"),
                "langchain": ("translation_plugin_langchain", "LangChainTranslateService"),
            }

            module_name, class_name = registry.get(provider, registry["etranslation"])
            module_path = f"..{module_name}"
            logger.info(f"Loading translation module: {module_path}, class: {class_name}, provider: {provider}")
            module = importlib.import_module(module_path, package=__package__)
            service_cls = getattr(module, class_name)
            service = service_cls()
            self._translator = Translator(services_list=[service])

            logger.info("Translator initialized successfully")

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
                GRAPH """ + sparql_escape_uri(GRAPHS["ai"]) + """ {
                    ?ann oa:hasBody ?stmt .
                    ?stmt a rdf:Statement ;
                           rdf:subject $source ;
                           rdf:predicate eli:language ;
                           rdf:object ?language .
                }
            }
            LIMIT 1
            """).substitute(source=sparql_escape_uri(self.source))

        result = query(query_string, sudo=True)
        bindings = result.get("results", {}).get("bindings", [])
        language_uri = None
        if bindings and "language" in bindings[0] and "value" in bindings[0].get("language", {}):
            language_uri = bindings[0]["language"]["value"]

        if language_uri:
            lang_code = LANGUAGE_URI_TO_CODE.get(language_uri)
            if lang_code:
                logger.info(
                    f"Retrieved source language from database: {lang_code}")
                return lang_code
            logger.warning(f"Unknown language URI: {language_uri}")

        # Fallback: detect language from content
        logger.warning(
            "No language found in database, detecting from content...")
        if not content:
            content = self.fetch_data()

        if not content or not content.strip():
            raise ValueError(
                f"No content available for language detection: {self.source}")

        # Limit text length for language detection to avoid hanging on very long text
        # langdetect can be slow/hang on very long text, so use first 1000 chars
        text_for_detection = content[:1000] if len(content) > 1000 else content
        logger.debug(
            f"Using {len(text_for_detection)} chars (of {len(content)}) for language detection")

        try:
            detected_lang = langdetect.detect(text_for_detection)
            logger.info(
                f"Detected language from content: {detected_lang}")
            return detected_lang
        except Exception as e:
            raise RuntimeError(
                f"Language detection failed for {self.source} "
                f"(content length={len(content)}): {e}"
            ) from e

    def create_language_relation(self, source_uri: str, language: str):
        """
        Create a language annotation for the source expression.
        """
        lang_uri = LANGUAGE_CODE_TO_URI.get(language)
        if not lang_uri:
            error_msg = f"Unsupported language code '{language}' - cannot create language annotation"
            logger.error(error_msg)
            raise ValueError(error_msg)
        RelationExtractionAnnotation(
            subject=source_uri,
            predicate="eli:language",
            obj=sparql_escape_uri(lang_uri),
            activity_id=self.task_uri,
            source_uri=source_uri,
            start=None,
            end=None,
            agent=get_agent_uri("translator"),
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
        translated_expr_uri = f"{SPARQL_PREFIXES["expressions"]}{uuid4()}"
        target_graph_uri = graph_uri or GRAPHS["ai"]

        realizes_triple = ""
        is_realized_by_triple = ""
        if work_uri:
            realizes_triple = f"{sparql_escape_uri(translated_expr_uri)} eli:realizes {sparql_escape_uri(work_uri)} ."
            is_realized_by_triple = f"{sparql_escape_uri(work_uri)} eli:is_realized_by {sparql_escape_uri(translated_expr_uri)} ."

        translation_triple = f"{sparql_escape_uri(source_expression_uri)} <{SPARQL_PREFIXES['linguistics_translations']}> {sparql_escape_uri(translated_expr_uri)} ."

        query_string = (
            get_prefixes_for_query("eli", "epvoc", "mu") +
            f"""
            INSERT {{
              GRAPH {sparql_escape_uri(target_graph_uri)} {{
                {sparql_escape_uri(translated_expr_uri)} a eli:Expression ;
                    mu:uuid "{str(uuid.uuid4())}" ;
                    epvoc:expressionContent {sparql_escape_string(translated_text)} .
                {realizes_triple}
                {is_realized_by_triple}
                {translation_triple}
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
                agent=get_agent_uri("translator"),
                agent_type=AGENT_TYPES["ai_component"],
            ).add_to_triplestore_if_not_exists()

        logger.info(
            f"Created translated expression {translated_expr_uri} in graph {target_graph_uri} (language: {target_language})")
        return translated_expr_uri

    def get_target_graph(self) -> str | None:
        """Try to resolve an optional target graph from the job linked to this task."""
        q = Template(
            """
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
            SELECT ?graph WHERE {
                $task dct:isPartOf ?job .
                ?job ext:graphForTargets ?graph .
            }
            """
        ).substitute(task=sparql_escape_uri(self.task_uri))
        res = query(q, sudo=True)
        bindings = res.get("results", {}).get("bindings", [])
        if not bindings:
            return None
        return bindings[0]["graph"]["value"]

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
        target_graph = self.get_target_graph()
        expressions_graph = sparql_escape_uri(target_graph if target_graph else GRAPHS["expressions"])
        works_graph = sparql_escape_uri(target_graph if target_graph else GRAPHS["works"])

        q = Template(
            get_prefixes_for_query("task", "epvoc", "eli") +
            f"""
            SELECT ?expression ?content ?lang ?work WHERE {{
            GRAPH {sparql_escape_uri(GRAPHS["jobs"])} {{
                $task task:inputContainer ?container .
            }}

            GRAPH {sparql_escape_uri(GRAPHS["data_containers"])} {{
                ?container task:hasResource ?expression .
            }}

            GRAPH {expressions_graph} {{
                ?expression a eli:Expression ;
                            epvoc:expressionContent ?content ;
                            eli:language ?lang .
            }}

            GRAPH {works_graph} {{
                ?work a eli:Work ;
                    eli:is_realized_by ?expression .
            }}
            }}
            """
        ).substitute(task=sparql_escape_uri(self.task_uri))

        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        if not bindings:
            logger.warning(
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
            GRAPH {sparql_escape_uri(GRAPHS["data_containers"])} {{
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
            logger.info(
                f"Processing translation for source: {source_expression_uri}")

            # If there's no content, skip early
            if not original_text or not original_text.strip():
                logger.warning(
                    "No content found for source; skipping translation")
                return

            # Ensure we have an explicit source language
            if not source_language or source_language.lower() == "auto":
                logger.warning(
                    "Source language was 'auto' or missing, detecting from content...")
                # Limit text length for language detection to avoid hanging on very long text
                text_for_detection = original_text[:1000] if len(
                    original_text) > 1000 else original_text
                logger.debug(
                    f"Using {len(text_for_detection)} chars (of {len(original_text)}) for language detection")
                try:
                    source_language = langdetect.detect(text_for_detection)
                except Exception as e:
                    raise RuntimeError(
                        f"Language detection failed for {source_expression_uri} "
                        f"(content length={len(original_text)}): {e}"
                    ) from e
                logger.info(
                    f"Detected source language: {source_language}")

            # Skip translation if already in target language
            if source_language.lower() == self.target_language.lower():
                logger.info(
                    f"Text is already in target language ({self.target_language}), skipping translation")
                return

            # Get translator and translate
            translator = self.get_translator()

            logger.info(
                f"Translating from {source_language} to {self.target_language}")

            # Use translatepy's translate method
            try:
                translation_result = translator.translate(
                    original_text,
                    destination_language=self.target_language,
                    source_language=source_language
                )
            except Exception as e:
                raise RuntimeError(
                    f"Translation failed for {source_expression_uri} "
                    f"({source_language} -> {self.target_language}): {e}"
                ) from e

            # Extract the translated text from the result
            translated_text = translation_result.result
            service_used = translation_result.service
            logger.info(f"Translation service used: {service_used}")

            logger.info(f"Translation completed.")

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

            logger.info(
                f"Translation stored as new expression: {translated_expression_uri}")

            self.results_container_uris.append(
                self.create_output_container(translated_expression_uri))
