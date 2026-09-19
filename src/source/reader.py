"""Generic, spec-driven :class:`SourceReader`.

The single reader implementation for the whole service. It knows nothing about
ELI, OSLO, or any other ontology — it builds its SPARQL from a
:class:`~src.source.base.SourceSpec` (a small JSON file of IRIs). The spec is the
per-ontology "constants": the resource classes, the text/language predicates, and
how the work/anchor is resolved. Add a new ontology by adding a new spec file; no
code change.

All IRIs are emitted fully-qualified (no ``PREFIX`` declarations), so the reader
has no dependency on the shared ``SPARQL_PREFIXES`` table knowing an ontology.
Runtime URIs (the task URI, a resource URI) and named-graph IRIs are escaped via
``sparql_escape_uri``; spec IRIs are trusted config authored with the service.
"""

from typing import Optional

from helpers import query
from escape_helpers import sparql_escape_uri

from decide_ai_service_base.sparql_config import (
    GRAPHS,
    LANGUAGE_URI_TO_CODE,
    SPARQL_PREFIXES,
)

from .base import Source, SourceReader, SourceSpec, Shape


# Stable infrastructure IRIs (the task/input vocabulary), derived from the base
# package so they track the environment if the prefix is ever reconfigured.
_INPUT_CONTAINER = sparql_escape_uri(SPARQL_PREFIXES["task"] + "inputContainer")
_HAS_RESOURCE = sparql_escape_uri(SPARQL_PREFIXES["task"] + "hasResource")


def _to_sources(bindings: list[dict]) -> list[Source]:
    """Bind SPARQL results to :class:`Source` objects, one per resource URI.

    A first-wins de-duplication on ``uri`` yields a single source per resource
    even if the OPTIONAL language/work patterns were to multiply rows.
    """
    sources: dict[str, Source] = {}
    for b in bindings:
        uri = b.get("e", {}).get("value")
        if not uri or uri in sources:
            continue
        sources[uri] = Source(
            uri=uri,
            content=b.get("content", {}).get("value", ""),
            language=LANGUAGE_URI_TO_CODE.get(b.get("lang", {}).get("value")),
            work_uri=b.get("work", {}).get("value"),
        )
    return list(sources.values())


class GenericSourceReader(SourceReader):
    """Read a task's input sources, parameterised entirely by a :class:`SourceSpec`."""

    def __init__(self, spec: SourceSpec):
        self._spec = spec

    # -- SPARQL fragment builders ------------------------------------------ #

    def _work_clause(self, shape: Shape) -> str:
        w = shape.work
        if w.mode == "self":
            return "BIND(?e AS ?work)"
        if w.mode == "predicate":
            assert w.predicate
            prop = sparql_escape_uri(w.predicate)
            if w.inverse:
                prop = f"{prop} | ^{sparql_escape_uri(w.inverse)}"
            # Work is an anchor, not content: keep the source even when the link
            # is absent, so the predicate is wrapped in an OPTIONAL.
            return f"OPTIONAL {{ ?e {prop} ?work }}"
        return ""

    def _branch(self, shape: Shape, content_graph: str) -> str:
        content = (
            f"GRAPH {content_graph} {{ "
            f"VALUES ?cls {{ {' '.join(sparql_escape_uri(c) for c in shape.classes)} }} . "
            "?e a ?cls . "
            f"VALUES ?tp {{ {' '.join(sparql_escape_uri(t) for t in shape.text)} }} . "
            "?e ?tp ?content . "
        )
        if shape.language:
            content += f"OPTIONAL {{ ?e {sparql_escape_uri(shape.language)} ?lang }} . "
        content += "}"
        work = self._work_clause(shape)
        return "{ " + content + (" " + work if work else "") + " }"

    # -- SourceReader interface -------------------------------------------- #

    def fetch_sources(self, task_uri: str, target_graph: Optional[str] = None) -> list[Source]:
        content_graph = sparql_escape_uri(target_graph) if target_graph else "?g"
        union = " UNION ".join(self._branch(s, content_graph) for s in self._spec.shapes)
        q = (
            "SELECT DISTINCT ?e ?content ?lang ?work WHERE { "
            f"GRAPH {sparql_escape_uri(GRAPHS['jobs'])} {{ {sparql_escape_uri(task_uri)} {_INPUT_CONTAINER} ?c }} "
            f"GRAPH {sparql_escape_uri(GRAPHS['data_containers'])} {{ ?c {_HAS_RESOURCE} ?e }} "
            f"{union} "
            "}"
        )
        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        return _to_sources(bindings)

    def get_source_text(self, uri: str) -> str:
        s = sparql_escape_uri(uri)
        q = (
            "SELECT DISTINCT ?content WHERE { "
            "GRAPH ?g { "
            f"VALUES ?tp {{ {' '.join(sparql_escape_uri(t) for t in self._spec.text_predicates)} }} . "
            f"{s} ?tp ?content "
            "} }"
        )
        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        # ?content is a required pattern here, so any returned row has text bound.
        return bindings[0]["content"].get("value", "") if bindings else ""

    def resolve_work(self, uri: str) -> Optional[str]:
        clauses = []
        s = sparql_escape_uri(uri)
        for shape in self._spec.shapes:
            w = shape.work
            if w.mode == "self":
                clauses.append(
                    "{ "
                    f"VALUES ?cls {{ {' '.join(sparql_escape_uri(c) for c in shape.classes)} }} . "
                    f"{s} a ?cls . "
                    f"BIND({s} AS ?work) "
                    "}"
                )
            elif w.mode == "predicate":
                assert w.predicate
                prop = sparql_escape_uri(w.predicate)
                if w.inverse:
                    prop = f"{prop} | ^{sparql_escape_uri(w.inverse)}"
                clauses.append(f"{{ {s} {prop} ?work }}")
        if not clauses:
            return None
        q = "SELECT ?work WHERE { " + " UNION ".join(clauses) + " } LIMIT 1"
        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        return bindings[0]["work"]["value"] if bindings else None
