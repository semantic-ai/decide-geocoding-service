import uuid
from typing import Optional, Any

from helpers import query, update
from string import Template
from escape_helpers import sparql_escape_uri, sparql_escape_string

from decide_ai_service_base.task import DecisionTask
from decide_ai_service_base.sparql_config import get_prefixes_for_query, GRAPHS, TASK_OPERATIONS, AI_COMPONENTS, AGENT_TYPES, LANGUAGE_CODE_TO_URI, LANGUAGE_URI_TO_CODE
from decide_ai_service_base.annotation import RelationExtractionAnnotation

from ..ner_functions import extract_entities
from ..config import get_config
from ..entity_mappers import map_entity_to_annotations, resolve_work_uri_for_expression
from ..library.entity_projections import project_spans
from ..library.entity_formatter import EntityFormatter

# Instantiate a single EntityFormatter to reuse across tasks.
entity_formatter = EntityFormatter()

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

    def create_general_entity_annotations(self, source_uri: str, entities: list[dict[str, Any]]) -> list[str]:
        """
        Creates RDF annotations for a list of general entity extraction results using per-label handler logic.

        Args:
            source_uri (str): The URI of the source expression or resource.
            entities (list[dict[str, Any]]): List of extracted entities, each being a dictionary
                with at minimum 'label', 'text', 'start', and 'end' keys.

        Returns:
            list[str]: List of URIs of the created annotation resources for applicable entities.
        """
        print(f"[GENERAL ENTITIES] subject: {self.source}; source_uri: {source_uri}")

        created_annotation_uris: list[str] = []

        # Resolve the work URI for this expression once (may be None if unknown)
        work_uri = resolve_work_uri_for_expression(source_uri)

        for entity in entities:
            try:
                annotation_uris = map_entity_to_annotations(
                    task=self,
                    work_uri=work_uri,
                    expression_uri=source_uri,
                    entity=entity,
                )
                created_annotation_uris.extend(annotation_uris)
            except Exception as e:
                self.logger.error(
                    f"Failed to map entity '{entity.get('text')}' "
                    f"({entity.get('label')}) for source {source_uri}: {e}",
                    exc_info=True,
                )

        return created_annotation_uris

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
            GRAPH {sparql_escape_uri(GRAPHS["jobs"])} {{
                $task task:inputContainer ?container .
            }}

            GRAPH {sparql_escape_uri(GRAPHS["data_containers"])} {{
                ?container task:hasResource ?expression .
            }}

            GRAPH {sparql_escape_uri(GRAPHS["expressions"])} {{
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
            
            # Clean up entities by formatting the dates, periods and splitting the locations into individual entities.
            general_entities_formatted = entity_formatter.format(general_entities)
            print(f"[FORMATTED ENTITIES] {general_entities_formatted}")

            entity_uris = self.create_general_entity_annotations(
                target_expression_uri, general_entities_formatted)

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

            # Format the projected entities as well to ensure the same level of detail in the annotations (e.g. date parsing, location splitting).
            general_entities_projected_formatted = entity_formatter.format(general_entities_projected)
            entity_uris_projected = self.create_general_entity_annotations(source_expression_uri, general_entities_projected_formatted)


            for projected_entity_uri in entity_uris_projected:
                self.results_container_uris.append(
                    self.create_output_container(projected_entity_uri))
