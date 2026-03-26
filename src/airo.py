from string import Template
from helpers import update
from escape_helpers import sparql_escape_uri
from decide_ai_service_base.sparql_config import get_prefixes_for_query, GRAPHS, ORGANIZATIONS


def register_airo():
    """Register the DECIDe AI system and its components in the triplestore."""
    digiteam = ORGANIZATIONS["digiteam"]
    query_template = Template(
        get_prefixes_for_query("mu", "foaf", "airo", "example", "prov", "lblod") +
        """
    INSERT DATA{
        GRAPH $graph {
            example:DECIDe a airo:AISystem ;
                airo:isDevelopedBy $provider ;
                airo:hasComponent example:entity-extraction .
                
            $provider a airo:AIDeveloper .
            
            example:entity-extraction a lblod:AIComponent .
        }
    }
    """)
    query_string = query_template.substitute(
        provider=sparql_escape_uri(digiteam),
        graph=sparql_escape_uri(GRAPHS["ai"])
    )
    update(query_string, sudo=True)
