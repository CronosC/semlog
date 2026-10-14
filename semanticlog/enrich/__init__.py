"""Optional analyses over a log.

`DeclareEnricher` mines DECLARE constraints and needs the Lydia solver, which
is not installed with the package. `DeclareResult` does not need it.
"""

from semanticlog.enrich.declare import DeclareEnricher, DeclareResult

__all__ = ["DeclareEnricher", "DeclareResult"]
