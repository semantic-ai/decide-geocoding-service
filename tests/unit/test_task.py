from src.task.base import Task


class A(Task):
    __task_type__ = "http://example.org/A"


class B(A):
    __task_type__ = "http://example.org/B"


class C(Task):
    pass


class D(Task):
    __task_type__ = "http://example.org/D"


def test_lookup():
    assert Task.lookup("http://example.org/A") == A
    assert Task.lookup("http://example.org/B") == B
    assert Task.lookup("http://example.org/C") is None
    assert Task.lookup("http://example.org/D") is D