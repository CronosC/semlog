"""Where the axioms of a build go.

`InMemorySink` keeps an ontology object for in-process reasoning and
inspection. `StreamingSink` writes axioms to disk as they arrive.
"""

from semlog.onto.sinks.base import AxiomSink
from semlog.onto.sinks.memory import InMemorySink
from semlog.onto.sinks.streaming import StreamingSink

__all__ = ["AxiomSink", "InMemorySink", "StreamingSink"]
