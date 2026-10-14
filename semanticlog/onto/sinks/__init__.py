"""Where the axioms of a build go.

`InMemorySink` keeps an ontology object for in-process reasoning and
inspection. `StreamingSink` writes axioms to disk as they arrive.
"""

from semanticlog.onto.sinks.base import AxiomSink
from semanticlog.onto.sinks.memory import InMemorySink
from semanticlog.onto.sinks.streaming import StreamingSink

__all__ = ["AxiomSink", "InMemorySink", "StreamingSink"]
