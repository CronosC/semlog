"""The emitted ontology against its recorded baseline.

A change to what a build emits fails here. The README in the golden
directory describes how to re-baseline.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import DEFAULT_IRI, OntoBuildSettings
from semanticlog.onto.sinks import StreamingSink
from tests.onto.support.configurations import CONFIGURATIONS, build_store

GOLDEN_DIR = Path(__file__).parent / "golden"

_VERSION_LINE = re.compile(r'"semlog [^"]*"')

_BUILD_CONFIG_LINE = re.compile(
    r"(AnnotationAssertion\(<[^>]*#buildConfiguration> <[^>]*> ).*"
)

_NOT_AN_AXIOM = ("Prefix(", "Ontology(", ")")

_REBASELINED: list[str] = []


def _normalize(line: str) -> str:
    line = _VERSION_LINE.sub('"semlog <version>"', line)
    return _BUILD_CONFIG_LINE.sub(r"\1<settings>)", line)


def build_axioms(settings: OntoBuildSettings, tmp_path: Path) -> list[str]:
    return sorted(_normalize(line) for line in build_lines(settings, tmp_path))


def build_lines(settings: OntoBuildSettings, tmp_path: Path) -> list[str]:
    target = tmp_path / "golden.ofn"
    sink = StreamingSink(str(target), DEFAULT_IRI, settings)
    OCEDOntoBuilder(DEFAULT_IRI, settings, sink=sink).build_ontology(
        build_store(), source="golden"
    )
    return [
        line
        for line in target.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(_NOT_AN_AXIOM)
    ]


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_build_output_matches_the_committed_baseline(name, tmp_path):
    axioms = build_axioms(CONFIGURATIONS[name](), tmp_path)
    fixture = GOLDEN_DIR / f"{name}.ofn"

    if os.environ.get("SEMLOG_UPDATE_GOLDEN"):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        fixture.write_text("\n".join(axioms) + "\n", encoding="utf-8")
        _REBASELINED.append(fixture.name)
        return

    assert fixture.is_file(), (
        f"no baseline for {name!r}. Create it with "
        "SEMLOG_UPDATE_GOLDEN=1 pytest tests/onto/test_golden.py"
    )
    expected = fixture.read_text(encoding="utf-8").splitlines()
    assert axioms == expected, (
        f"{name}: build output no longer matches {fixture}. If the change is "
        "intended, re-baseline with SEMLOG_UPDATE_GOLDEN=1 and say why in the "
        "commit message."
    )


def test_a_rebaselining_run_does_not_report_green():
    if not os.environ.get("SEMLOG_UPDATE_GOLDEN"):
        return
    pytest.fail(
        f"SEMLOG_UPDATE_GOLDEN was set and {len(_REBASELINED)} fixture(s) were "
        f"rewritten: {', '.join(sorted(_REBASELINED))}. Review the diff, then "
        "re-run without the variable."
    )


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_the_build_emits_no_duplicate_axioms(name, tmp_path):
    from collections import Counter

    counts = Counter(build_lines(CONFIGURATIONS[name](), tmp_path))
    repeated = {line: n for line, n in counts.items() if n > 1}
    assert not repeated, "{} repeated axiom(s):\n{}".format(
        len(repeated),
        "\n".join(f"  {n}x  {line}" for line, n in sorted(repeated.items())),
    )


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_the_build_configuration_annotation_is_written_and_parses(name, tmp_path):
    written = [
        line
        for line in build_lines(CONFIGURATIONS[name](), tmp_path)
        if "#buildConfiguration>" in line
    ]
    assert len(written) == 1, (
        f"expected one buildConfiguration annotation, got {len(written)}"
    )

    payload = written[0].split('"', 1)[1].rsplit('"', 1)[0]
    settings = json.loads(payload.replace('\\"', '"').replace("\\\\", "\\"))
    assert "profile" in settings
    assert "order_representations" in settings


def test_a_value_outside_every_bin_is_not_emitted_twice(tmp_path):
    from collections import Counter

    from semanticlog.onto.settings import DataRepresentation
    from tests.onto.support.configurations import out_of_bin_store, settings

    configuration = settings(
        data_representations=[DataRepresentation.VALUES, DataRepresentation.DIST_BINS],
        order_representations=[],
        case_concepts=[],
    )
    target = tmp_path / "out_of_bin.ofn"
    sink = StreamingSink(str(target), DEFAULT_IRI, configuration)
    OCEDOntoBuilder(DEFAULT_IRI, configuration, sink=sink).build_ontology(
        out_of_bin_store()
    )

    lines = [
        line
        for line in target.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(_NOT_AN_AXIOM)
    ]
    repeated = {line: n for line, n in Counter(lines).items() if n > 1}
    assert not repeated, "{} repeated axiom(s):\n{}".format(
        len(repeated),
        "\n".join(f"  {n}x  {line}" for line, n in sorted(repeated.items())),
    )
