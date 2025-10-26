# SPARQL Config - Usage Reference

## Import
```python
from .sparql_config import (
    get_prefixes_for_query,    # Generate PREFIX declarations
    GRAPHS,                     # Graph URIs: "ai", "jobs"
    ORGANIZATIONS,              # "digiteam"
    JOB_STATUSES,               # "scheduled", "busy", "success", "failed"
    TASK_OPERATIONS,            # "entity_extraction"
    AGENT_TYPES,                # "person", "ai_component"
    AI_COMPONENTS,              # "ner_extractor", "decide_system"
    ONTOLOGY_CLASSES,           # "location", "address", "annotation", etc.
)
```

## Usage Examples

### Generate Prefixes
```python
prefixes = get_prefixes_for_query("oa", "prov", "mu")
query = Template(prefixes + "SELECT ?x WHERE { ?x a oa:Annotation . }")
```

### Use Constants
```python
graph = GRAPHS["ai"]                           # http://mu.semte.ch/graphs/ai
status = JOB_STATUSES["busy"]                  # http://redpencil.data.gift/id/concept/JobStatus/busy
agent = AGENT_TYPES["person"]                  # http://www.w3.org/ns/prov#Person
component = AI_COMPONENTS["ner_extractor"]     # http://example.org/entity-extraction
```

## Adding Constants
Edit `sparql_config.py` and add to appropriate section, then import and use:
```python
# In sparql_config.py
SPARQL_PREFIXES["my_ns"] = "https://example.org/my-namespace#"

# Then import and use
prefixes = get_prefixes_for_query("my_ns")
```
