"""Where the axioms of a build go.

`InMemorySink` keeps an ontology object, which in-process reasoning and
inspection need. `StreamingSink` writes to a file as it goes and holds almost
nothing, for large logs. Both are held to the same conformance tests, so a
build produces the same ontology either way.
"""

from semanticlog.onto.sinks.base import OWL_DATATYPES, AxiomSink, normalize_value
from semanticlog.onto.sinks.memory import InMemorySink
from semanticlog.onto.sinks.streaming import StreamingSink

__all__ = [
    "OWL_DATATYPES",
    "AxiomSink",
    "InMemorySink",
    "StreamingSink",
    "normalize_value",
]
