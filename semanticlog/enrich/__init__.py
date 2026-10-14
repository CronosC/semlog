"""Optional analyses over a log.

`DeclareEnricher` mines DECLARE constraints and needs the Lydia solver, which
is not installed with the package. The result it returns, `DeclareResult`,
lives in `semanticlog.data.declare`.
"""

from semanticlog.enrich.declare import DeclareEnricher

__all__ = ["DeclareEnricher"]
