import pytest
from helpers import query

@pytest.fixture(autouse=True)
def clear_graph():
    yield
    query("CLEAR GRAPH <http://mu.semte.ch/graphs/ai>")
