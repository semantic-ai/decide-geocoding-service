from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from helpers import update, query
from escape_helpers import sparql_escape_uri, sparql_escape_string, sparql_escape_date

from decide_ai_service_base.sparql_config import get_prefixes_for_query, GRAPHS, AI_COMPONENTS, AGENT_TYPES, SPARQL_PREFIXES
from decide_ai_service_base.annotation import RelationExtractionAnnotation


def _parse_date_literal(text: str) -> str:
    """Convert a date string into a typed xsd:date literal.

    Args:
        text: Raw date text extracted by NER, e.g. 17.09.2021 or 17 September 2021.

    Returns:
        A SPARQL-ready literal string. When parsing succeeds this is of the
        form YYYY-MM-DD^^xsd:date. When parsing fails, this is the
        original value wrapped as a normal string literal via sparql_escape_string.
    """
    value = text.strip()
    if not value:
        return sparql_escape_string(value)

    # Handle bare month + year, e.g. "September 2021" or "Sep 2021" by normalising to the first day of that month.
    month_names = {
        "JANUARY": "01",
        "FEBRUARY": "02",
        "MARCH": "03",
        "APRIL": "04",
        "MAY": "05",
        "JUNE": "06",
        "JULY": "07",
        "AUGUST": "08",
        "SEPTEMBER": "09",
        "OCTOBER": "10",
        "NOVEMBER": "11",
        "DECEMBER": "12",
    }
    parts = value.replace(",", " ").split()
    if len(parts) == 2:
        month, year = parts
        key = month.upper()
        # Support both full and short forms (e.g. "Sep")
        for full_name, num in month_names.items():
            if key == full_name or key == full_name[:3]:
                if year.isdigit() and len(year) == 4:
                    return sparql_escape_date(datetime(int(year), int(num), 1).date())

    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d %B %Y", "%d %b %Y"):
        try:
            dt = datetime.strptime(value, fmt)
            return sparql_escape_date(dt.date())
        except ValueError:
            continue

    # Fallback: plain literal.
    return sparql_escape_string(value)


def resolve_work_uri_for_expression(expression_uri: str) -> Optional[str]:
    """Resolve the eli:Work realized by an expression.

    Args:
        expression_uri: URI of the expression whose work we want to resolve.

    Returns:
        The work URI if a matching ?expr eli:realizes ?work triple is
        found in any graph, otherwise None.
    """
    q = (
        get_prefixes_for_query("eli")
        + """
        SELECT ?work WHERE {
          GRAPH ?g {
            $expr eli:realizes ?work .
          }
        }
        LIMIT 1
        """
    )

    res = query(q.replace("$expr", sparql_escape_uri(expression_uri)), sudo=True)
    bindings = res.get("results", {}).get("bindings", [])
    if bindings and "work" in bindings[0]:
        return bindings[0]["work"]["value"]
    return None


def _insert_triples(insert_body: str) -> None:
    """Insert triples into the AI graph."""
    q = (
        get_prefixes_for_query(
            "foaf", "dct", "dcterms", "rdfs", "skos", "eli", "eli-dl", "prov", "mu", "besluit"
        )
        + f"""
        INSERT DATA {{
          GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
{insert_body}
          }}
        }}
        """
    )
    update(q, sudo=True)


def _annotate(task,subject: str, predicate: str, obj: str, source_uri: str, entity: Dict[str, Any]) -> str:
    """Create a TripletAnnotation and return its URI."""
    return RelationExtractionAnnotation(
        subject=subject,
        predicate=predicate,
        obj=obj,
        activity_id=task.task_uri,
        source_uri=source_uri,
        start=entity.get("start"),
        end=entity.get("end"),
        agent=AI_COMPONENTS["ner_extractor"],
        agent_type=AGENT_TYPES["ai_component"],
        confidence=entity.get("confidence", 1.0),
    ).add_to_triplestore_if_not_exists()


def _create_work_date_annotation(task, work_uri: Optional[str], source_uri: str, entity: Dict[str, Any], predicate_uri: str) -> List[str]:
    """Helper to create a single date statement on the work."""
    obj_literal = _parse_date_literal(entity.get("text", ""))
    ann_uri = _annotate(
        task,
        subject=work_uri,
        predicate=predicate_uri,
        obj=obj_literal,
        source_uri=source_uri,
        entity=entity,
    )
    return [ann_uri]


def _get_or_create_session_for_work(task, work_uri: Optional[str]) -> Optional[str]:
    """Return a session Activity URI associated with a work, creating if needed.

    This ensures that all SESSION_DATE / MANDATARY entities for the
    same work within a single task execution share one eli-dl:Activity
    node, typed as a besluit:Zitting activity.

    Args:
        task: The running EntityExtractionTask instance. A small cache is
            stored on the task instance to reuse the session per work.
        work_uri: URI of the eli:Work to which the session relates.

    Returns:
        A session URI (https://data.lblod.info/id/session/...) or None when work_uri is missing.
    """
    if not work_uri:
        return None

    cache: Dict[str, str] = getattr(task, "_entity_mapper_session_cache", {})
    task._entity_mapper_session_cache = cache

    if work_uri in cache:
        return cache[work_uri]

    session_uuid = uuid.uuid4()
    session_uri = f"https://data.lblod.info/id/session/{session_uuid}"

    insert_body = f"""
    {sparql_escape_uri(session_uri)} a eli-dl:Activity ;
        mu:uuid "{str(session_uuid)}" ;
        eli-dl:had_activity_type besluit:Zitting .
"""
    _insert_triples(insert_body)

    cache[work_uri] = session_uri
    return session_uri


def map_entity_to_annotations(task, work_uri: Optional[str], expression_uri: str, entity: Dict[str, Any]) -> List[str]:
    """Map a refined entity into one or more RDF annotations.

    This is the main entry point used by EntityExtractionTask. It examines
    the entity's refined label and emits the appropriate Open Annotation +
    rdf:Statement structure in the AI/public graph.

    The function is deliberately side-effectful (it writes to the triplestore)
    and returns the URIs of the created annotations so callers can attach them to the task's result containers.

    Args:
        task: The active EntityExtractionTask instance; provides task_uri for provenance and logging.
        work_uri: Optional URI of the decision eli:Work related to the expression. Some labels (e.g. dates, legal grounds) require this to be present; when it's missing, those labels become no-ops.
        expression_uri: URI of the expression where the entity was detected. This becomes the oa:hasSource in the annotation's target.
        entity: Dict representing a single NER entity, typically with keys text, label, start, end and confidence.

    Returns:
        A list of annotation URIs (strings) that were created for this entity.
        It may be empty when the label is not handled or required context is
        missing (e.g. no work URI).
    """
    label = str(entity.get("label", "")).upper()

    created: List[str] = []

    # All currently supported labels represent properties or relations on the
    # decision Work (or on resources that are only meaningful in the context of
    # that Work, such as sessions, normative provisions, or locations). If we
    # cannot resolve a Work URI, we deliberately no-op for these labels to avoid
    # emitting half‑connected RDF.
    work_bound_labels = {
        "PUBLICATION_DATE",
        "LEGAL_DATE",
        "CONTEXT_DATE",
        "ENTRY_DATE",
        "EXPIRY_DATE",
        "CONTEXT_PERIOD",
        "VALIDITY_PERIOD",
        "SESSION_DATE",
        "MANDATARY",
        "LEGAL_GROUNDS",
        "ADMINISTRATIVE_BODY",
        "CONTEXT_LOCATION",
        "IMPACT_LOCATION",
    }
    if not work_uri and label in work_bound_labels:
        return created

    # Publication / legal / context dates on the work.
    if label == "PUBLICATION_DATE":
        created += _create_work_date_annotation(task, work_uri, expression_uri, entity, "eli:publication_date")
        return created

    if label == "LEGAL_DATE":
        created += _create_work_date_annotation(task, work_uri, expression_uri, entity, "eli:date_document")
        return created

    if label == "CONTEXT_DATE":
        created += _create_work_date_annotation(task, work_uri, expression_uri, entity, "dct:date")
        return created

    if label == "ENTRY_DATE":
        # Map to both first_date_entry_in_force and date_applicability.
        created += _create_work_date_annotation(task, work_uri, expression_uri, entity, "eli:first_date_entry_in_force")
        created += _create_work_date_annotation(task, work_uri, expression_uri, entity, "eli:date_applicability")
        return created

    if label == "EXPIRY_DATE":
        created += _create_work_date_annotation(task, work_uri, expression_uri, entity, "eli:date_no_longer_in_force")
        created += _create_work_date_annotation(task, work_uri, expression_uri, entity, "eli:date_applicability")
        return created

    # Context period: create a ProperInterval node and link from work via dct:date
    if label == "CONTEXT_PERIOD":
        period_id = str(uuid.uuid4())
        period_uri = f"https://data.lblod.info/id/period/{period_id}"
        begin_literal = _parse_date_literal(entity.get("text", ""))
        end_literal = begin_literal  # minimal: same date for start/end
        insert_body = f"""
    {sparql_escape_uri(period_uri)} a {sparql_escape_uri("http://www.w3.org/2006/time#ProperInterval")} ;
        mu:uuid "{period_id}" ;
        {sparql_escape_uri("http://www.w3.org/2006/time#hasBeginning")} {begin_literal} ;
        {sparql_escape_uri("http://www.w3.org/2006/time#hasEnd")} {end_literal} .
"""
        _insert_triples(insert_body)

        ann_uri = _annotate(
            task,
            subject=work_uri,
            predicate="dct:date",
            obj=sparql_escape_uri(period_uri),
            source_uri=expression_uri,
            entity=entity,
        )
        created.append(ann_uri)
        return created

    # Validity period: create Normatieve bepaling + ProperInterval and link from Work.
    if label == "VALIDITY_PERIOD":
        # Create a NormatieveBepaling with a dedicated period resource and
        # map the first date of the period as an eli:date_applicability on
        # the Work (see specification examples).
        period_id = str(uuid.uuid4())
        nb_id = str(uuid.uuid4())
        period_uri = f"https://data.lblod.info/id/period/{period_id}"
        nb_uri = f"https://data.lblod.info/id/normatievebepaling/{nb_id}"
        begin_literal = _parse_date_literal(entity.get("text", ""))
        end_literal = begin_literal
        insert_body = f"""
    {sparql_escape_uri(period_uri)} a {sparql_escape_uri("http://www.w3.org/2006/time#ProperInterval")} ;
        {sparql_escape_uri("http://www.w3.org/2006/time#hasBeginning")} {begin_literal} ;
        {sparql_escape_uri("http://www.w3.org/2006/time#hasEnd")} {end_literal} ;
        mu:uuid "{period_id}" .

    {sparql_escape_uri(nb_uri)} a {sparql_escape_uri("https://data.vlaanderen.be/ns/omgevingsvergunning#NormatieveBepaling")} ;
        mu:uuid "{nb_id}" ;
        dct:extent {sparql_escape_uri(period_uri)} .
"""
        _insert_triples(insert_body)

        ann1 = _annotate(
            task,
            subject=nb_uri,
            predicate="dct:extent",
            obj=sparql_escape_uri(period_uri),
            source_uri=expression_uri,
            entity=entity,
        )
        created.append(ann1)

        ann2 = _annotate(
            task,
            subject=work_uri,
            predicate="eli:realizes",
            obj=sparql_escape_uri(nb_uri),
            source_uri=expression_uri,
            entity=entity,
        )
        created.append(ann2)
        # Additionally expose the first day of the validity period as
        # eli:date_applicability on the Work.
        created += _create_work_date_annotation(
            task, work_uri, expression_uri, entity, "eli:date_applicability"
        )
        return created

    # Session date: create a session Activity and link it to the work.
    if label == "SESSION_DATE":
        session_uri = _get_or_create_session_for_work(task, work_uri)
        # If the session URI is not found, return an empty list.
        if not session_uri:
            return created

        date_literal = _parse_date_literal(entity.get("text", ""))
        # Activity date
        ann1 = _annotate(
            task,
            subject=session_uri,
            predicate="eli-dl:activity_date",
            obj=date_literal,
            source_uri=expression_uri,
            entity=entity,
        )
        created.append(ann1)

        return created

    # Mandatary / participant person during session.
    if label == "MANDATARY":
        session_uri = _get_or_create_session_for_work(task, work_uri)
        # If the session URI is not found, return an empty list.
        if not session_uri:
            return created

        person_uuid = uuid.uuid4()
        person_uri = f"{SPARQL_PREFIXES["people"]}{person_uuid}"
        name_literal = sparql_escape_string(entity.get("text", ""))
        insert_body = f"""
            {sparql_escape_uri(person_uri)} a foaf:Person ;
            mu:uuid "{str(person_uuid)}" ;
            rdfs:label {name_literal} .
        """
        _insert_triples(insert_body)

        ann1 = _annotate(
            task,
            subject=session_uri,
            predicate="eli-dl:had_participant_person",
            obj=sparql_escape_uri(person_uri),
            source_uri=expression_uri,
            entity=entity,
        )
        created.append(ann1)

        return created

    # Legal grounds: cited work with label.
    if label == "LEGAL_GROUNDS":
        cited_uuid = uuid.uuid4()
        cited_uri = f"{SPARQL_PREFIXES["legal_expressions"]}{cited_uuid}"
        label_literal = sparql_escape_string(entity.get("text", ""))
        insert_body = f"""
    {sparql_escape_uri(cited_uri)} a eli:Work ;
        mu:uuid "{str(cited_uuid)}" ;
        rdfs:label {label_literal} .
"""
        _insert_triples(insert_body)

        ann = _annotate(
            task,
            subject=work_uri,
            predicate="eli:cites",
            obj=sparql_escape_uri(cited_uri),
            source_uri=expression_uri,
            entity=entity,
        )
        created.append(ann)
        return created

    # Administrative body: organization that passed the decision.
    if label == "ADMINISTRATIVE_BODY":
        org_uuid = uuid.uuid4()
        org_uri = f"{SPARQL_PREFIXES["organizations"]}{org_uuid}"
        label_literal = sparql_escape_string(entity.get("text", ""))
        insert_body = f"""
    {sparql_escape_uri(org_uri)} a {sparql_escape_uri("http://www.w3.org/ns/org#Organization")} ;
        mu:uuid "{str(org_uuid)}" ;
        rdfs:label {label_literal} .
"""
        _insert_triples(insert_body)

        ann = _annotate(
            task,
            subject=work_uri,
            predicate="eli:passed_by",
            obj=sparql_escape_uri(org_uri),
            source_uri=expression_uri,
            entity=entity,
        )
        created.append(ann)
        return created

    # Context location: general spatial context for the work.
    if label == "CONTEXT_LOCATION":
        loc_uuid = uuid.uuid4()
        loc_uri = f"{SPARQL_PREFIXES["locations"]}{loc_uuid}"
        label_literal = sparql_escape_string(entity.get("text", ""))
        insert_body = f"""
    {sparql_escape_uri(loc_uri)} a dct:Location ;
        mu:uuid "{str(loc_uuid)}" ;
        rdfs:label {label_literal} .
"""
        _insert_triples(insert_body)

        ann = _annotate(
            task,
            subject=work_uri,
            predicate="dct:spatial",
            obj=sparql_escape_uri(loc_uri),
            source_uri=expression_uri,
            entity=entity,
        )
        created.append(ann)
        return created

    # Impact location: where the decision has effect.
    if label == "IMPACT_LOCATION":
        loc_uuid = uuid.uuid4()
        loc_uri = f"{SPARQL_PREFIXES["locations"]}{loc_uuid}"
        label_literal = sparql_escape_string(entity.get("text", ""))
        insert_body = f"""
    {sparql_escape_uri(loc_uri)} a dct:Location ;
        mu:uuid "{str(loc_uuid)}" ;
        rdfs:label {label_literal} .
"""
        _insert_triples(insert_body)

        ann = _annotate(
            task,
            subject=work_uri,
            predicate="prov:atLocation",
            obj=sparql_escape_uri(loc_uri),
            source_uri=expression_uri,
            entity=entity,
        )
        created.append(ann)
        return created

    # For now, other labels fall back to no-op; they can be added here later.
    return created