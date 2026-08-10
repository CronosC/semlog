from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass
from typing import cast

from owlapy.owl_ontology import Ontology

from semanticlog.onto.settings import OntoBuildSettings


@dataclass
class SemanticLog:
    """The output of an ontology build: the OWL ontology paired with its build settings.

    Attributes:
        onto: The built owlapy Ontology object, or ``None`` when the build streamed
            straight to disk and never held one.
        settings: The OntoBuildSettings used to produce this ontology.
        iri: The normalised ontology IRI (always ends with ``#``).
        path: Where a streamed build wrote its ontology. ``None`` for an in-memory
            build, whose output is not on disk until it is saved.
    """

    onto: Ontology | None
    settings: OntoBuildSettings
    iri: str
    path: str | None = None

    def require_onto(self) -> Ontology:
        """Return the in-memory ontology, or explain why there isn't one.

        Raises:
            ValueError: If the build streamed to disk. Nothing was held in memory,
                so the file is the only copy.
        """
        if self.onto is None:
            raise ValueError(
                "This ontology was streamed to "
                f"{self.path!r} and was never held in memory. Read it back from "
                "that file, or build with the default in-memory sink."
            )
        return self.onto

    # ------------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------------

    def reason(self, path: str | None = None, backend: str = "HermiT"):
        """Return a :class:`~semanticlog.onto.reason.Reasoner` over this ontology.

        The reasoner reads from a file, so an in-memory build has to be written
        out first. Doing that here keeps the round-trip in one place instead of
        every caller repeating it.

        Args:
            path: Where the ontology should live on disk. Defaults to :attr:`path`
                for a streamed build. Required for an in-memory build, because
                :meth:`~semanticlog.onto.reason.Reasoner.materialize` writes back
                to this location and a temporary file would discard the result.
            backend: Reasoner backend: ``"HermiT"``, ``"Pellet"`` or ``"ELK"``.

        Returns:
            A reasoner loaded from ``path``.

        Raises:
            ValueError: If no path is available to reason over.
        """
        from semanticlog.onto.reason import Reasoner

        target = path or self.path
        if target is None:
            raise ValueError(
                "This ontology is only in memory, so there is nothing for the "
                "reasoner to load. Pass reason(path=...) to say where it should be "
                "written; materialised results are saved back to that file."
            )
        if path is not None and self.onto is not None:
            self.write_ontology(path)
        return Reasoner(target, backend=backend, profile=self.settings.profile)

    def check(self, backend: str = "HermiT") -> bool:
        """Return True if the ontology is consistent.

        Unlike :meth:`reason`, this needs no destination: nothing is written back,
        so an in-memory ontology goes through a temporary file that is cleaned up.
        """
        if self.path is not None:
            return self.reason(backend=backend).check()

        tmp = tempfile.NamedTemporaryFile(suffix=".owl", delete=False)
        tmp.close()
        try:
            self.write_ontology(tmp.name)
            return self.reason(path=tmp.name, backend=backend).check()
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def write_ontology(self, path: str) -> None:
        """Write the OWL file to disk without the build settings.

        Use this when only the ontology file is needed (e.g. for loading into a
        reasoner or triple store).  For a round-trippable snapshot that also
        preserves settings, use :meth:`write` / :meth:`read` instead.

        Args:
            path: Destination file path (e.g. ``output.owl``).
        """
        self.require_onto().save(path)

    def write(self, path: str) -> None:
        """Write a pickle file containing the full SemanticLog.

        The file is typically given a ``.semlog`` extension.  The ontology
        is serialised to OWL/XML bytes internally so that the pickle works
        regardless of whether the owlapy Ontology is natively picklable.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def read(cls, path: str) -> SemanticLog:
        """Read a SemanticLog from a pickle file produced by write()."""
        with open(path, "rb") as f:
            return cast("SemanticLog", pickle.load(f))  # noqa: S301

    # ------------------------------------------------------------------
    # Pickle support: convert onto to/from OWL bytes
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        onto_bytes = self._onto_to_bytes()
        return {
            "onto_bytes": onto_bytes,
            "settings": self.settings,
            "iri": self.iri,
        }

    def __setstate__(self, state: dict) -> None:
        self.settings = state["settings"]
        self.iri = state["iri"]
        self.onto = self._onto_from_bytes(state["onto_bytes"])

    def _onto_to_bytes(self) -> bytes:
        onto = self.require_onto()
        tmp = tempfile.NamedTemporaryFile(suffix=".owl", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            onto.save(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def _onto_from_bytes(data: bytes) -> Ontology:
        from owlapy.iri import IRI

        tmp = tempfile.NamedTemporaryFile(suffix=".owl", delete=False)
        tmp_path = tmp.name
        try:
            tmp.write(data)
            tmp.close()
            return Ontology(IRI.create(tmp_path), load=True)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
