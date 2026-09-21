"""Ontology-agnostic source-reading seam.

Tasks read their input exclusively through :data:`reader` (a
:class:`~src.source.reader.GenericSourceReader`). That reader is parameterised
entirely by a small JSON **spec** of IRIs (``spec.json``) — the per-ontology
"constants": the source classes, the text/language predicates and how the
work/anchor is resolved. The service code never names an ontology.

The reader is built from the spec at ``config/spec.json`` (currently the OSLO
spec; the ELI reference spec lives at ``src/source/spec.json``). Switching
ontology is a *data* change — no code, no image rebuild: replace
``config/spec.json``, or volume-mount a different spec over it::

    - ./eli-spec.json:/app/config/spec.json:ro

Adding a further ontology is just authoring another ``spec.json``.

    from src.source import reader
    sources = reader.fetch_sources(task_uri)
"""

import os

from .base import Source, SourceReader, SourceSpec, load_spec
from .reader import GenericSourceReader


def build_reader(spec_path: str) -> GenericSourceReader:
    """Build a :class:`GenericSourceReader` from the spec at ``spec_path``."""   
    return GenericSourceReader(load_spec(spec_path))


# Built once at import. ``query``/``update`` only hit the triplestore when a
# method is actually called, so import-time construction is cheap and safe.
reader: SourceReader = build_reader(os.path.join('/','config', 'spec.json'))


def get_reader() -> SourceReader:
    """Return the active :class:`SourceReader` singleton."""
    return reader


__all__ = [
    "Source",
    "SourceReader",
    "SourceSpec",
    "GenericSourceReader",
    "load_spec",
    "build_reader",
    "reader",
    "get_reader",
]
