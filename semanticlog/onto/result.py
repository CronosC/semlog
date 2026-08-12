"""What a build hands back.

The result carries the ontology together with the settings that produced it,
so that later steps such as reasoning do not have to be told the profile
again. A streamed build has no ontology in memory and carries the path it was
written to instead.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

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
    """

    onto: Ontology | None
    settings: OntoBuildSettings
    iri: str
    path: str | None = None

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

    def reason(self, path: str | None = None, backend: str = "HermiT"):
        """Return a reasoner over this ontology.

        A reasoner reads from a file, so an in-memory ontology has to be
        written before it can be reasoned over. Passing a path writes it there
        first, and anything the reasoner materialises is saved back to that
        same file.

        Args:
            path: Where to write the ontology first. Defaults to the path a
                streamed build already wrote to.
            backend: Reasoner to use.

        Returns:
            A reasoner ready to check or materialize.

        Raises:
            ValueError: If the ontology is only in memory and no path is given.
        """
        from semanticlog.onto.reason import Reasoner

        target = path or self.path
        if target is None:
            raise ValueError(
                "This ontology is only in memory, so there is nothing for the "
                "reasoner to load. Pass reason(path=...) to say where it should be "
                "written; materialized results are saved back to that file."
            )
        if path is not None and self.onto is not None:
            self.save(path)
        return Reasoner(target, backend=backend, profile=self.settings.profile)

    def check(self, backend: str = "HermiT") -> bool:
        """Report whether the ontology is consistent.

        An in-memory ontology is written to a temporary file for the reasoner
        to read, and the file is removed afterwards.

        Args:
            backend: Reasoner to use.

        Returns:
            True if the ontology is consistent.
        """
        if self.path is not None:
            return self.reason(backend=backend).check()

        tmp = tempfile.NamedTemporaryFile(suffix=".owl", delete=False)
        tmp.close()
        try:
            self.save(tmp.name)
            return self.reason(path=tmp.name, backend=backend).check()
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    def save(self, path: str) -> None:
        """Save the ontology to a file.

        Raises:
            ValueError: If the build streamed and kept no ontology to save.
        """
        self.require_onto().save(path)
