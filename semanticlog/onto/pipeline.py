"""One build, from a cleaned store to an ontology file.

`build_log` picks the sink for the output format, runs `OCEDOntoBuilder` and
writes the file.

    from semanticlog.data.flatten import choose_case_concepts
    from semanticlog.data.loading import load_and_clean
    from semanticlog.onto.inputs import BuildInputs
    from semanticlog.onto.pipeline import build_log
    from semanticlog.onto.settings import OntoBuildSettings

    data = load_and_clean("log.jsonocel")
    case_concepts, _ = choose_case_concepts(data, None)
    build_log(data, OntoBuildSettings(), BuildInputs(case_concepts), "log.ofn")
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.readable import ReadableStore
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.result import BuildResult
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink, InMemorySink, StreamingSink


def build_log(
    data: ReadableStore,
    settings: OntoBuildSettings,
    inputs: BuildInputs,
    output: str | Path,
    *,
    iri: str = DEFAULT_IRI,
    source: str | None = None,
    record: Mapping[str, Any] | None = None,
    streaming: bool = True,
) -> BuildResult:
    """Build one cleaned store into an ontology file and return the result.

    Args:
        data: Store to build from. Must be cleaned.
        inputs: Case concepts, DECLARE constraints and annotations resolved
            against this log.
        output: Path of the ontology file. Its directory is created. The file
            is written when the build completes.
        source: Path of the log, recorded as provenance.
        record: Other values to record about this run, such as the input
            checksum and the command. Each key must be a safe IRI local name.
        streaming: Write functional syntax to the file as axioms are emitted.
            False builds the ontology in memory and writes RDF/XML.

    Returns:
        The finished build. `onto` is set for an in-memory build. `path` is
        always set.

    Raises:
        ConfigError: If the base IRI, the settings or the inputs are unusable.
        InputError: If the store fails a pre-build check, or a name in it
            cannot be emitted.
        ValueError: If a key of `record` cannot be an IRI local name.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sink: AxiomSink = (
        StreamingSink(str(output), iri, settings.profile, settings.tbox_iri)
        if streaming
        else InMemorySink(iri, settings.profile, settings.tbox_iri)
    )
    result = OCEDOntoBuilder(iri, settings, sink=sink).build_ontology(
        data, inputs, source=source, run=record
    )
    sink.save(str(output))
    if result.path is None:
        result = dataclasses.replace(result, path=str(output))
    return result
