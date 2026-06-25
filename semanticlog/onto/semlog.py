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
        onto: The built owlapy Ontology object.
        settings: The OntoBuildSettings used to produce this ontology.
        iri: The normalised ontology IRI (always ends with ``#``).
    """

    onto: Ontology
    settings: OntoBuildSettings
    iri: str

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
        self.onto.save(path)

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
        tmp = tempfile.NamedTemporaryFile(suffix=".owl", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            self.onto.save(tmp_path)
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
