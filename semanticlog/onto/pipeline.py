"""One build, from a cleaned store to an ontology file.

`build_log` picks the sink for the suffix of the output path, runs
`OCEDOntoBuilder` and writes the file.

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
from pathlib import Path

from semanticlog.core.enums import OutputFormat
from semanticlog.core.errors import ConfigError
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
) -> BuildResult:
    """Build one cleaned store into an ontology file and return the result.

    Args:
        data: Store to build from. Must be cleaned.
        inputs: Case concepts, DECLARE constraints and annotations resolved
            against this log.
        output: Path of the ontology file. Its directory is created. The file
            is written when the build completes. A `.ofn` path streams
            functional syntax to the file as axioms are emitted. A `.owl` path
            builds the ontology in memory and writes RDF/XML.

    Returns:
        The finished build. `onto` is set for an in-memory build. `path` is
        always set.

    Raises:
        ConfigError: If the output path ends in neither `.ofn` nor `.owl`, or
            the base IRI, the settings or the inputs are unusable.
        InputError: If the store fails a pre-build check, or a name in it
            cannot be emitted.
    """
    output = Path(output)
    output_format = _build_format(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sink: AxiomSink = (
        StreamingSink(str(output), iri, settings.profile, settings.tbox_iri)
        if output_format is OutputFormat.OFN
        else InMemorySink(iri, settings.profile, settings.tbox_iri)
    )
    result = OCEDOntoBuilder(iri, settings, sink=sink).build_ontology(data, inputs)
    sink.save(str(output))
    if result.path is None:
        result = dataclasses.replace(result, path=str(output))
    return result


def _build_format(output: Path) -> OutputFormat:
    """Return the format a build writes to `output`.

    Raises:
        ConfigError: If the suffix is neither `.ofn` nor `.owl`.
    """
    suffix = output.suffix.casefold()
    if suffix == ".ofn":
        return OutputFormat.OFN
    if suffix == ".owl":
        return OutputFormat.OWL
    raise ConfigError(
        f"Cannot build into {output}. Give a path ending in .ofn for functional "
        "syntax or in .owl for RDF/XML."
    )
