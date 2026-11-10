"""The result of a build.

`BuildResult` holds the ontology, or the path of a streamed build, with the
settings that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from owlapy.owl_ontology import Ontology

from semanticlog.onto.settings import OntoBuildSettings


@dataclass
class BuildResult:
    """The ontology a build produced and the settings it ran with.

    Attributes:
        onto: The ontology, or None when the build streamed it to a file.
        settings: Settings the build ran with.
        iri: Base IRI of the ontology.
        path: Where the ontology was written. `build_log` always sets it. None
            for an in-memory build run through `OCEDOntoBuilder` directly.
        stats: How many classes, properties and axioms it holds. Both sinks
            report the same keys.
    """

    onto: Ontology | None
    settings: OntoBuildSettings
    iri: str
    path: str | None = None
    stats: dict[str, int] = field(default_factory=dict)

    def require_onto(self) -> Ontology:
        """Return the ontology.

        Raises:
            ValueError: If the build streamed to a file.
        """
        if self.onto is None:
            raise ValueError(
                f"This ontology was streamed to {self.path!r} and is not in "
                "memory. Read it from that file, or build with streaming off."
            )
        return self.onto
