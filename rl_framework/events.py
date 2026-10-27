"""Optional structured events; ordinary CLI/Python usage has no event sink."""
from contextlib import contextmanager
from contextvars import ContextVar

_sink = ContextVar("run_event_sink", default=None)
_context = ContextVar("run_event_context", default={})


@contextmanager
def event_sink(callback):
    token = _sink.set(callback)
    try:
        yield
    finally:
        _sink.reset(token)


@contextmanager
def event_context(**fields):
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def emit(kind, **fields):
    if callback := _sink.get():
        callback({**_context.get(), **fields, "type": kind})
