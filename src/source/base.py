"""Ontology-agnostic source-reading contract + spec model.

The triplestore *reading* of a task's input is isolated behind :class:`SourceReader`
so the rest of the service (segmentation, NER, translation, annotation) stays
ontology-neutral: it operates on whatever :class:`Source` the active reader
returns, and never needs to know whether the underlying resource is an ELI
expression, an OSLO besluit, or anything else.

A reader is fully parameterised by a :class:`SourceSpec` (a small JSON file of
IRIs). The single generic implementation is :class:`src.source.reader.GenericSourceReader`;
switching ontology is a *data* change — replace ``config/spec.json`` (or volume-mount a
different spec over it). No code change, no image rebuild.
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Source:
    """A single source resource read from a task's input container.

    Ontology-neutral: ``uri`` may be an ``eli:Expression``, an ``oslo:Besluit``,
    or any other resource the active reader returns. Downstream code treats it
    opaquely (annotations, segments and translations are anchored on it).
    """

    uri: str
    content: str
    language: Optional[str] = None
    work_uri: Optional[str] = None


# --------------------------------------------------------------------------- #
# Spec model — the per-ontology "constants" that the generic reader adapts to.
# --------------------------------------------------------------------------- #

@dataclass
class WorkSpec:
    """How a source's *work*/anchor is resolved.

    mode ``self``      -> the work is the source resource itself (OSLO besluit).
    mode ``predicate`` -> the work is found via ``predicate`` (and optionally its
                          ``inverse``) from the source (ELI: ``eli:realizes``).
    mode ``none``      -> no work; ``work_uri`` is always ``None``.
    """

    mode: str = "none"
    predicate: Optional[str] = None
    inverse: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "WorkSpec":
        mode = d.get("mode", "none")
        if mode not in ("self", "predicate", "none"):
            raise ValueError(f"work.mode must be one of 'self', 'predicate', 'none', got {mode!r}")
        predicate = d.get("predicate")
        inverse = d.get("inverse")
        if mode == "predicate" and not predicate:
            raise ValueError("work.mode 'predicate' requires a non-empty 'predicate' IRI")
        return cls(mode=mode, predicate=predicate, inverse=inverse)


@dataclass
class Shape:
    """One way a source resource can appear in the store.

    A spec is a list of shapes; the reader UNIONs over them and matches whichever
    shape is actually present in a task's input container — a shape that doesn't
    match a given container is simply inert. ELI lists one shape
    (``eli:Expression``). OSLO lists two: the original ``oslo:Besluit`` (read by the
    translation step) and the ``eli:Expression`` artifact that translation emits
    (read by downstream segmentation/NER). When the translation step is skipped,
    the artifact shape never matches and downstream reads the besluiten directly.
    """

    classes: list[str]
    text: list[str]
    language: Optional[str] = None
    work: WorkSpec = field(default_factory=WorkSpec)

    @classmethod
    def from_dict(cls, d: dict) -> "Shape":
        classes = d.get("classes") or []
        text = d.get("text") or []
        if not classes:
            raise ValueError("shape requires a non-empty 'classes' list of IRIs")
        if not text:
            raise ValueError("shape requires a non-empty 'text' list of predicate IRIs")
        return cls(
            classes=list(classes),
            text=list(text),
            language=d.get("language"),
            work=WorkSpec.from_dict(d.get("work") or {}),
        )


@dataclass
class SourceSpec:
    """The full per-ontology configuration consumed by the generic reader."""

    shapes: list[Shape]

    @classmethod
    def from_dict(cls, d: dict) -> "SourceSpec":
        shapes = d.get("shapes")
        if not isinstance(shapes, list) or not shapes:
            raise ValueError("spec requires a non-empty 'shapes' list")
        return cls(shapes=[Shape.from_dict(s) for s in shapes])

    @classmethod
    def from_file(cls, path: str) -> "SourceSpec":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @property
    def text_predicates(self) -> list[str]:
        """Deduped union of every shape's text predicates (order preserved)."""
        seen, out = set(), []
        for shape in self.shapes:
            for p in shape.text:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        return out

    @property
    def language_predicates(self) -> list[str]:
        """Deduped union of every shape's language predicates (order preserved)."""
        seen, out = set(), []
        for shape in self.shapes:
            if shape.language and shape.language not in seen:
                seen.add(shape.language)
                out.append(shape.language)
        return out


def load_spec(path: str) -> SourceSpec:
    """Load and validate a :class:`SourceSpec` from a JSON file at ``path``."""
    if not os.path.exists(path):
        raise ValueError(f"source spec not found: {path}")
    return SourceSpec.from_file(path)


# --------------------------------------------------------------------------- #
# Reader contract
# --------------------------------------------------------------------------- #

class SourceReader(ABC):
    """Contract for reading a task's input sources from the triplestore."""

    @abstractmethod
    def fetch_sources(self, task_uri: str, target_graph: Optional[str] = None) -> list[Source]:
        """Return the sources referenced by ``task_uri``'s input container.

        ``target_graph`` optionally scopes the resources' content to a named
        graph (used by the translation task to honour a job-level target graph);
        when ``None`` the content is read from any graph.
        """

    @abstractmethod
    def get_source_text(self, uri: str) -> str:
        """Return the free text of a single source resource."""

    @abstractmethod
    def resolve_work(self, uri: str) -> Optional[str]:
        """Resolve the work/anchor a source belongs to (may be ``None``)."""
