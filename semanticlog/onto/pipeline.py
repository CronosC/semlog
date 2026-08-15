"""One build, from a cleaned store to an ontology file.

`OCEDOntoBuilder` produces axioms and `AxiomSink` writes them, and between the
two sits a handful of decisions every build makes the same way: which sink the
output format asks for, where the file goes, and that it appears there only
once it is complete. Those live here rather than in the command line, so that
building from Python is the same build the CLI runs.

    from semanticlog.data.flatten import choose_case_concepts
    from semanticlog.data.loading import load_and_clean
    from semanticlog.onto.pipeline import build_log

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
    """Build one cleaned store into an ontology file.

    Args:
        data: Store to build from, already cleaned.
        settings: What to build.
        inputs: Case concepts, DECLARE constraints and annotations resolved
            against this log.
        output: Where the ontology belongs. Its directory is created, and the
            file appears there only once the build is complete.
        iri: Base IRI for the ontology.
        source: Path of the log, recorded as provenance.
        record: Anything else to record about this run, such as the input
            checksum and the command.
        streaming: Write functional syntax straight to the file. Off builds
            the ontology in memory and writes RDF/XML.

    Returns:
        The finished build, carrying the ontology when it was built in memory
        and the path either way.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sink: AxiomSink = (
        StreamingSink(str(output), iri, settings.profile)
        if streaming
        else InMemorySink(iri, settings.profile)
    )
    result = OCEDOntoBuilder(iri, settings, sink=sink).build_ontology(
        data, inputs, source=source, run=record
    )
    sink.save(str(output))
    if result.path is None:
        result = dataclasses.replace(result, path=str(output))
    return result
