"""
SPARQL Configuration and Constants

This module centralizes all SPARQL prefixes, URIs, and constants used throughout
the codebase. By maintaining these in one place, updates to URIs or prefixes only
need to be made once, reducing maintenance burden and preventing inconsistencies.
"""

# ==============================================================================
# SPARQL NAMESPACE PREFIXES
# ==============================================================================
# Maps prefix names to their full URIs for use in SPARQL queries

SPARQL_PREFIXES = {
    "mu": "http://mu.semte.ch/vocabularies/core/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "airo": "https://w3id.org/airo#",
    "example": "http://www.example.org/",
    "ex": "http://example.org/",
    "prov": "http://www.w3.org/ns/prov#",
    "lblod": "https://data.vlaanderen.be/ns/lblod#",
    "oa": "http://www.w3.org/ns/oa#",
    "dct": "http://purl.org/dc/terms/",
    "dcterms": "http://purl.org/dc/terms/",
    "skolem": "http://www.example.org/id/.well-known/genid/",
    "nif": "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#",
    "locn": "http://www.w3.org/ns/locn#",
    "geosparql": "http://www.opengis.net/ont/geosparql#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "adms": "http://www.w3.org/ns/adms#",
    "task": "http://lblod.data.gift/vocabularies/tasks/",
    "nfo": "http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#",
    "eli": "http://data.europa.eu/eli/ontology#",
    "eli-dl": "http://data.europa.eu/eli/eli-dl#",
    "epvoc": "https://data.europarl.europa.eu/def/epvoc#",
    "ns1": "http://www.w3.org/ns/dqv#",
    "ns2": "https://w3id.org/okn/o/sd#",
    "ns3": "https://w3id.org/airo#",
    "schema": "https://schema.org/",
}

# ==============================================================================
# GRAPH URIs
# ==============================================================================
# Named graphs in the RDF store

GRAPHS = {
    "ai": "http://mu.semte.ch/graphs/ai",
    "jobs": "http://mu.semte.ch/graphs/jobs",
    "oparl_temp": "http://mu.semte.ch/graphs/oparl-temp",
    "oslo_temp": "http://mu.semte.ch/graphs/oslo-temp",
}

# ==============================================================================
# ORGANIZATION & AGENT URIs
# ==============================================================================

ORGANIZATIONS = {
    "digiteam": "https://www.vlaanderen.be/organisaties/administratieve-diensten-van-de-vlaamse-overheid/beleidsdomein-kanselarij-bestuur-buitenlandse-zaken-en-justitie/agentschap-binnenlands-bestuur/digiteam",
}

# ==============================================================================
# JOB STATUS URIs
# ==============================================================================

JOB_STATUS_BASE = "http://redpencil.data.gift/id/concept/JobStatus"

JOB_STATUSES = {
    "scheduled": f"{JOB_STATUS_BASE}/scheduled",
    "busy": f"{JOB_STATUS_BASE}/busy",
    "success": f"{JOB_STATUS_BASE}/success",
    "failed": f"{JOB_STATUS_BASE}/failed",
}

# ==============================================================================
# TASK OPERATION URIs
# ==============================================================================

TASK_OPERATIONS = {
    "entity_extraction": "http://lblod.data.gift/id/jobs/concept/TaskOperation/entity-extracting",
    "model_annotation": "http://lblod.data.gift/id/jobs/concept/TaskOperation/model-annotating",
    "model_batch_annotation": "http://lblod.data.gift/id/jobs/concept/TaskOperation/model-batch-annotating",
    "classifier_training": "http://lblod.data.gift/id/jobs/concept/TaskOperation/classifier-training",
    "geo_extraction": "http://lblod.data.gift/id/jobs/concept/TaskOperation/geo-extracting",
    "translation": "http://lblod.data.gift/id/jobs/concept/TaskOperation/translating",
    "segmentation": "http://lblod.data.gift/id/jobs/concept/TaskOperation/segmenting",
}

# ==============================================================================
# AGENT TYPES
# ==============================================================================

AGENT_TYPES = {
    "person": "http://www.w3.org/ns/prov#Person",
    "ai_component": "https://data.vlaanderen.be/ns/lblod#AIComponent",
}

# ==============================================================================
# AI COMPONENT URIs
# ==============================================================================

AI_COMPONENTS = {
    "ner_extractor": "http://example.org/entity-extraction",
    "model_annotater": "http://example.org/model_annotation",
    "decide_system": "http://example.org/DECIDe",
    "translator": "http://example.org/translation",
    "segmenter": "http://example.org/segmentation",
}

# ==============================================================================
# LANGUAGE MAPPINGS
# ==============================================================================

LANGUAGE_CODE_TO_URI = {
    'nl': "http://publications.europa.eu/resource/authority/language/NLD",
    'de': "http://publications.europa.eu/resource/authority/language/DEU",
    'en': "http://publications.europa.eu/resource/authority/language/ENG",
}

LANGUAGE_URI_TO_CODE = {
    "http://publications.europa.eu/resource/authority/language/NLD": "nl",
    "http://publications.europa.eu/resource/authority/language/DEU": "de",
    "http://publications.europa.eu/resource/authority/language/ENG": "en",
}

# ==============================================================================
# ONTOLOGY CLASS TYPES
# ==============================================================================

ONTOLOGY_CLASSES = {
    "location": "http://purl.org/dc/terms/Location",
    "address": "http://www.w3.org/ns/locn#Address",
    "street_name": "https://data.vlaanderen.be/ns/adres#Straatnaam",
    "annotation": "http://www.w3.org/ns/oa#Annotation",
    "ai_system": "https://w3id.org/airo#AISystem",
    "ai_developer": "https://w3id.org/airo#AIDeveloper",
    "ai_component": "https://data.vlaanderen.be/ns/lblod#AIComponent",
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_prefix_section() -> str:
    """
    Generate a complete SPARQL PREFIX section as a string.
    
    Returns:
        A string containing all PREFIX declarations suitable for prepending
        to SPARQL queries.
        
    Example:
        >>> query = get_prefix_section() + "SELECT ?s WHERE { ... }"
    """
    lines = ["PREFIX {0}: <{1}>".format(prefix, uri) 
             for prefix, uri in SPARQL_PREFIXES.items()]
    return "\n".join(lines) + "\n"


def get_prefixes_for_query(*prefix_names: str) -> str:
    """
    Generate a SPARQL PREFIX section for only the specified prefixes.
    
    Args:
        *prefix_names: Variable number of prefix names to include
        
    Returns:
        A string containing the requested PREFIX declarations
        
    Example:
        >>> query = get_prefixes_for_query("oa", "prov", "mu")
        >>> query += "SELECT ?s WHERE { ... }"
    """
    lines = []
    for prefix_name in prefix_names:
        if prefix_name in SPARQL_PREFIXES:
            uri = SPARQL_PREFIXES[prefix_name]
            lines.append("PREFIX {0}: <{1}>".format(prefix_name, uri))
    if not lines:
        raise ValueError(f"No valid prefixes found in: {prefix_names}")
    return "\n".join(lines) + "\n"
