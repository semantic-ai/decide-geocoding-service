import uuid
from typing import Optional, Any

from helpers import query, update
from string import Template
from escape_helpers import sparql_escape_uri, sparql_escape_string

from decide_ai_service_base.task import DecisionTask
from decide_ai_service_base.sparql_config import get_prefixes_for_query, GRAPHS, TASK_OPERATIONS, AI_COMPONENTS, AGENT_TYPES, LANGUAGE_CODE_TO_URI, LANGUAGE_URI_TO_CODE
from decide_ai_service_base.annotation import RelationExtractionAnnotation

from ..config import get_config
from ..library.entity_projections import project_spans


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
              GRAPH """ + sparql_escape_uri(GRAPHS["ai"]) + """ {
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
        from ..library.segmentors import GemmaSegmentor, LLMSegmentor
        seg_config = get_config().segmentation
        api_key = seg_config.llm.api_key.get_secret_value() if seg_config.llm.api_key else None

        if seg_config.llm.model_name == "wdmuer/decide-marked-segmentation":
            return GemmaSegmentor(
                api_key=api_key,
                base_url=seg_config.llm.base_url,
                model_name=seg_config.llm.model_name,
                temperature=seg_config.llm.temperature,
                max_new_tokens=seg_config.max_new_tokens,
            )
        else:
            return LLMSegmentor(
                provider=seg_config.llm.provider,
                model_name=seg_config.llm.model_name,
                api_key=api_key,
                base_url=seg_config.llm.base_url,
                temperature=seg_config.llm.temperature,
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

            # Use segment label as predicate (e.g., "ext:participants")
            # Convert to proper predicate format

            predicate = f"ext:{segment_label}"
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

            # Explicitly exclude title segments as this is already done in PDF content extraction service
            segments = [segment for segment in segments if segment["label"].lower() != "title"]
            self.logger.info(f"Excluded returned {len(segments)} segments, excluding titles")

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