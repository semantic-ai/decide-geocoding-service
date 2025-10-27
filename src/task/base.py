import contextlib
import logging

from abc import ABC, abstractmethod
from typing import Optional, Any

from string import Template
from helpers import query
from escape_helpers import sparql_escape_uri, sparql_escape_string
from ..sparql_config import get_prefixes_for_query, GRAPHS, JOB_STATUSES, TASK_OPERATIONS, AI_COMPONENTS, AGENT_TYPES


class Task(ABC):
    """Base class for background tasks that process data from the triplestore."""

    def __init__(self, task_uri: str):
        super().__init__()
        self.task_uri = task_uri
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def lookup(cls, task_type: str) -> Optional['Task']:
        """
        Yield all subclasses of the given class, per:
        https://adamj.eu/tech/2024/05/10/python-all-subclasses/
        """
        for subclass in cls.__subclasses__():
            if hasattr(subclass, '__task_type__') and subclass.__task_type__ == task_type:
                return subclass
            else:
                res = subclass.lookup(task_type)
                if res is not None:
                    return res
        return None

    @classmethod
    def from_uri(cls, task_uri: str) -> 'Task':
        """Create a Task instance from its URI in the triplestore."""
        q = Template(
            get_prefixes_for_query("adms", "task") +
            """
            SELECT ?task ?taskType WHERE {
              ?task task:operation ?taskType .
              FILTER(?task = $uri)
            }
        """).substitute(uri=sparql_escape_uri(task_uri))
        for b in query(q).get('results').get('bindings'):
            candidate_cls = cls.lookup(b['taskType']['value'])
            if candidate_cls is not None:
                return candidate_cls(task_uri)
            raise RuntimeError("Unknown task type {0}".format(b['taskType']['value']))
        raise RuntimeError("Task with uri {0} not found".format(task_uri))

    def change_state(self, old_state: str, new_state: str, results_container_uri: str = "") -> None:
        """Update the task status in the triplestore."""
        query_template = Template(
            get_prefixes_for_query("task", "adms") +
            """
            DELETE {
            GRAPH <""" + GRAPHS["jobs"] + """> {
                ?task adms:status ?oldStatus .
            }
            }
            INSERT {
            GRAPH <""" + GRAPHS["jobs"] + """> {
                ?task
                $results_container_line
                adms:status <$new_status> .

            }
            }
            WHERE {
            GRAPH <""" + GRAPHS["jobs"] + """> {
                BIND($task AS ?task)
                BIND(<$old_status> AS ?oldStatus)
                OPTIONAL { ?task adms:status ?oldStatus . }
            }
            }
            """)

        results_container_line = ""
        if results_container_uri:
            results_container_line = f"task:resultsContainer <{results_container_uri}> ;"

        query_string = query_template.substitute(
            new_status=JOB_STATUSES[new_state],
            old_status=JOB_STATUSES[old_state],
            task=sparql_escape_uri(self.task_uri),
            results_container_line=results_container_line)

        query(query_string)

    @contextlib.contextmanager
    def run(self):
        """Context manager for task execution with state transitions."""
        self.change_state("scheduled", "busy")
        yield
        self.change_state("busy", "success")

    def execute(self):
        """Run the task and handle state transitions."""
        with self.run():
            self.process()

    @abstractmethod
    def process(self):
        """Process task data (implemented by subclasses)."""
        pass


class DecisionTask(Task, ABC):
    """Task that processes decision-making data with input and output containers."""
    
    def __init__(self, task_uri: str):
        super().__init__(task_uri)

        q = Template(
            get_prefixes_for_query("dct", "task", "nfo") +
            """
        SELECT ?source WHERE {
          BIND($task AS ?t)
          ?t a task:Task .
          OPTIONAL { 
            ?t task:inputContainer ?ic . 
            OPTIONAL { ?ic a nfo:DataContainer ; task:hasResource ?source . }
          }
        }
        """).substitute(task=sparql_escape_uri(task_uri))
        r = query(q)
        self.source = r["results"]["bindings"][0]["source"]["value"]

    def fetch_data(self) -> str:
        """Retrieve the input data for this task from the triplestore."""
        query_string = f"""
        SELECT ?title ?description ?decision_basis WHERE {{
        BIND(<{self.source}> AS ?s)
        OPTIONAL {{ ?s <http://data.europa.eu/eli/ontology#title> ?title }}
        OPTIONAL {{ ?s <http://data.europa.eu/eli/ontology#description> ?description }}
        OPTIONAL {{ ?s <http://data.europa.eu/eli/eli-dl#decision_basis> ?decision_basis }}
        }}
        """

        query_result = query(query_string)

        title = query_result["results"]["bindings"][0]["title"]["value"]
        description = query_result["results"]["bindings"][0]["description"]["value"]
        decision_basis = query_result["results"]["bindings"][0]["decision_basis"]["value"]


        return "\n".join([title, description, decision_basis])

