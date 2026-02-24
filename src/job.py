from string import Template

from helpers import update
from escape_helpers import sparql_escape_uri

from .sparql_config import get_prefixes_for_query, GRAPHS, JOB_STATUSES, TASK_OPERATIONS, prefixed_log


from string import Template

from helpers import update
from escape_helpers import sparql_escape_uri
from .sparql_config import get_prefixes_for_query, GRAPHS, JOB_STATUSES, TASK_OPERATIONS, prefixed_log


def fail_busy_and_scheduled_tasks():
    """
    Fails all busy tasks for the given operations (or all if none provided).
    """
    prefixed_log("Startup: failing busy tasks if there are any")

    operations = TASK_OPERATIONS.values()

    # Build the VALUES clause dynamically
    operations_values = " ".join(sparql_escape_uri(op) for op in operations)

    q = Template(
        get_prefixes_for_query("task", "adms", "dct") +
        f"""
        DELETE {{
            GRAPH $graph {{
                ?task adms:status ?status .
            }}
        }}
        INSERT {{
            GRAPH $graph {{
                ?task adms:status {sparql_escape_uri(JOB_STATUSES["failed"])} .
            }}
        }}
        WHERE {{
            GRAPH $graph {{
                ?task a task:Task ;
                      dct:isPartOf ?job ;
                      task:operation ?operation ;
                      adms:status ?status .
                VALUES ?operation {{
                    {operations_values}
                }}
                VALUES ?status {{
                    {sparql_escape_uri(JOB_STATUSES["busy"])}
                }}
            }}
        }}
        """
    ).substitute(graph=sparql_escape_uri(GRAPHS["jobs"]))

    update(q, sudo=True)
