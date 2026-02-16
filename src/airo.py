from string import Template
from helpers import update
from escape_helpers import sparql_escape_uri
from .sparql_config import get_prefixes_for_query, GRAPHS, ORGANIZATIONS


def register_airo():
    """Register the DECIDe AI system and its components in the triplestore."""
    digiteam = ORGANIZATIONS["digiteam"]
    query_template = Template(
        get_prefixes_for_query("mu", "foaf", "airo", "example", "prov", "lblod") +
        """
    INSERT DATA{
        GRAPH <""" + GRAPHS["ai"] + """> {
            example:DECIDe a airo:AISystem ;
                airo:isDevelopedBy $provider ;
                airo:hasComponent example:entity-extraction .
                
            $provider a airo:AIDeveloper .
            
            example:entity-extraction a lblod:AIComponent .
        }
    }
    """)
    query_string = query_template.substitute(
        provider=sparql_escape_uri(digiteam)
    )
    update(query_string, sudo=True)
