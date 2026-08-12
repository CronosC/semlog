"""Readers that turn source event logs into an OCED store.

`PM4PyLoader` is the one to use for OCEL 2.0. `ClassicalLoader` covers flat
single-case logs, which have no object-centric structure of their own.
"""

from semanticlog.data.loading.base import Loader
from semanticlog.data.loading.classical_loader import ClassicalLoader
from semanticlog.data.loading.pm4py_loader import PM4PyLoader

__all__ = ["Loader", "ClassicalLoader", "PM4PyLoader"]
