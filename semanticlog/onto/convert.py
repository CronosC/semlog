"""Writing a built ontology in further formats through the OWL API."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jpype
from owlapy.owl_ontology import SyncOntology

from semanticlog.core.enums import OutputFormat
from semanticlog.core.errors import InputError
from semanticlog.onto.reason import start_jvm
from semanticlog.onto.sinks.base import PARTIAL_SUFFIX

logger = logging.getLogger(__name__)

# OWL API document format class of each output format, in
# `org.semanticweb.owlapi.formats`.
_DOCUMENT_FORMATS: dict[OutputFormat, str] = {
    OutputFormat.OFN: "FunctionalSyntaxDocumentFormat",
    OutputFormat.OWL: "RDFXMLDocumentFormat",
    OutputFormat.OWX: "OWLXMLDocumentFormat",
    OutputFormat.TTL: "TurtleDocumentFormat",
    OutputFormat.OMN: "ManchesterSyntaxDocumentFormat",
}


def convert(source: Path, targets: Mapping[OutputFormat, Path]) -> None:
    """Write the ontology in `source` once per target format.

    The source is loaded once. Each file is written under `PARTIAL_SUFFIX` and
    moved into place when complete. The targets written before a failure stay.

    Raises:
        DependencyError: If there is no JVM to run the OWL API on.
        InputError: If the OWL API cannot read the source or write a target.
            A target that fails leaves no file.
    """
    if not targets:
        return
    start_jvm()
    try:
        ontology = SyncOntology(str(source))
    except Exception as exc:
        raise InputError(
            f"The OWL API cannot read {source}:\n  {exc}\n"
            "Build the ontology again, or check that a build wrote the file."
        ) from exc
    manager = ontology.owlapi_manager
    owlapi_ontology = ontology.get_owlapi_ontology()
    source_format = manager.getOntologyFormat(owlapi_ontology)
    for output_format, target in targets.items():
        _save(manager, owlapi_ontology, source_format, output_format, target)
        logger.info("Ontology saved as %s: %s", output_format.value, target)


def _save(
    manager: Any,
    owlapi_ontology: Any,
    source_format: Any,
    output_format: OutputFormat,
    target: Path,
) -> None:
    """Write one target under the partial suffix, then move it into place."""
    document_format = jpype.JClass(
        f"org.semanticweb.owlapi.formats.{_DOCUMENT_FORMATS[output_format]}"
    )()
    if document_format.isPrefixOWLDocumentFormat() and (
        source_format is not None and source_format.isPrefixOWLDocumentFormat()
    ):
        document_format.copyPrefixesFrom(source_format.asPrefixOWLDocumentFormat())
    partial = Path(f"{target}{PARTIAL_SUFFIX}")
    iri = jpype.JClass("org.semanticweb.owlapi.model.IRI").create(
        jpype.JClass("java.io.File")(str(partial))
    )
    try:
        manager.saveOntology(owlapi_ontology, document_format, iri)
        partial.replace(target)
    except Exception as exc:
        raise InputError(
            f"Cannot write {target} as {output_format.value}:\n  {exc}"
        ) from exc
    finally:
        partial.unlink(missing_ok=True)
