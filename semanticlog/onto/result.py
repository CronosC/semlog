"""What a build hands back.

The result carries the ontology together with the settings that produced it.
A streamed build has no ontology in memory and carries the path it was written
to instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from owlapy.owl_ontology import Ontology

from semanticlog.onto.settings import OntoBuildSettings


@dataclass
class BuildResult:
    """The ontology a build produced, with the settings behind it.

    Attributes:
        onto: The ontology, or None when the build streamed it to a file.
        settings: Settings the build ran with.
        iri: Base IRI of the ontology.
        path: Where the ontology was written, for a streamed build.
        stats: How many classes, properties and axioms it holds. The counts
            come from the sink's name ledger, so a streamed build and an
            in-memory one report the same keys and two runs can be compared.
    """

    onto: Ontology | None
    settings: OntoBuildSettings
    iri: str
    path: str | None = None
    stats: dict[str, int] = field(default_factory=dict)

    def require_onto(self) -> Ontology:
        """Return the ontology, or explain why there is none.

        Raises:
            ValueError: If the build streamed to a file and kept nothing.
        """
        if self.onto is None:
            raise ValueError(
                "This ontology was streamed to "
                f"{self.path!r} and was never held in memory. Read it back from "
                "that file, or build with the default in-memory sink."
            )
        return self.onto

    def save(self, path: str) -> None:
        """Save the ontology to a file.

        Raises:
            ValueError: If the build streamed and kept no ontology to save.
        """
        self.require_onto().save(path)
