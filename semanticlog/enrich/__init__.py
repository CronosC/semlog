"""Optional analyses that add to a log without reshaping it.

The mining part needs a solver that is not installed with the package.
`DeclareResult`, which is the part a build uses, carries no such dependency.
"""

from semanticlog.enrich.declare import DeclareEnricher, DeclareResult

__all__ = ["DeclareEnricher", "DeclareResult"]
