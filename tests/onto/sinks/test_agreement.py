"""Both sinks produce the same ontology from the same build."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter

import pytest

from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import DEFAULT_IRI
from tests.onto.scenarios.test_golden import build_lines
from tests.onto.support.configurations import CONFIGURATIONS, build_store

_NS = DEFAULT_IRI.rstrip("#") + "#"
_OWL = "{http://www.w3.org/2002/07/owl#}"
_RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"

_REF = re.compile(r":((?:[^\s()\\]|\\.)+)")

_OFN_ANNOTATION = re.compile(
    r'^AnnotationAssertion\(rdfs:(label|comment) :(\S+) "(.*)"\)$'
)


def _streaming_signature(settings, tmp_path) -> Counter:
    counts: Counter = Counter()
    for line in build_lines(settings, tmp_path):
        if line.startswith("Declaration("):
            kind = line[len("Declaration(") :].split("(")[0]
            name = _REF.search(line).group(1).replace("\\", "")
            counts[(kind, name)] += 1
    return counts


def _in_memory_signature(settings, tmp_path) -> Counter:
    target = tmp_path / "memory.owl"
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.build_ontology(build_store(), source="golden")
    builder.sink.save(str(target))

    kinds = {
        f"{_OWL}Class": "Class",
        f"{_OWL}ObjectProperty": "ObjectProperty",
        f"{_OWL}DatatypeProperty": "DataProperty",
    }
    counts: Counter = Counter()
    for node in ET.parse(target).getroot():
        kind = kinds.get(node.tag)
        about = node.get(f"{_RDF}about")
        if kind is None or about is None:
            continue
        if about.startswith("#"):
            counts[(kind, about[1:])] += 1
        elif about.startswith(_NS):
            counts[(kind, about[len(_NS) :])] += 1
    return counts


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_both_backends_declare_the_same_entities(name, tmp_path):
    settings_factory = CONFIGURATIONS[name]

    streaming = _streaming_signature(settings_factory(), tmp_path)
    in_memory = _in_memory_signature(settings_factory(), tmp_path)

    disagreement = {
        "only StreamingSink": sorted(set(streaming) - set(in_memory))[:10],
        "only InMemorySink": sorted(set(in_memory) - set(streaming))[:10],
    }
    assert not any(disagreement.values()), (
        f"{name}: the two backends disagree.\n"
        + "\n".join(f"  {side}: {entries}" for side, entries in disagreement.items())
    )


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_neither_backend_declares_anything_twice(name, tmp_path):
    settings_factory = CONFIGURATIONS[name]

    for label, counts in (
        ("StreamingSink", _streaming_signature(settings_factory(), tmp_path)),
        ("InMemorySink", _in_memory_signature(settings_factory(), tmp_path)),
    ):
        repeated = {entity: n for entity, n in counts.items() if n > 1}
        assert not repeated, f"{name}/{label} declares {sorted(repeated)[:10]} twice"


def _streaming_annotations(settings, tmp_path) -> set[tuple[str, str, str]]:
    found = set()
    for line in build_lines(settings, tmp_path):
        match = _OFN_ANNOTATION.match(line)
        if match:
            kind, name, text = match.groups()
            found.add(
                (
                    name.replace("\\", ""),
                    kind,
                    text.replace('\\"', '"').replace("\\\\", "\\"),
                )
            )
    return found


def _in_memory_annotations(settings, tmp_path) -> set[tuple[str, str, str]]:
    target = tmp_path / "memory-annotations.owl"
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.build_ontology(build_store(), source="golden")
    builder.sink.save(str(target))

    rdfs = "{http://www.w3.org/2000/01/rdf-schema#}"
    found = set()
    for node in ET.parse(target).getroot():
        about = node.get(f"{_RDF}about")
        if about is None:
            continue
        name = about[1:] if about.startswith("#") else about.removeprefix(_NS)
        for child in node:
            for kind in ("label", "comment"):
                if child.tag == f"{rdfs}{kind}":
                    found.add((name, kind, child.text or ""))
    return found


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_both_backends_write_the_same_annotations(name, tmp_path):
    settings_factory = CONFIGURATIONS[name]

    streaming = _streaming_annotations(settings_factory(), tmp_path)
    in_memory = _in_memory_annotations(settings_factory(), tmp_path)

    disagreement = {
        "only StreamingSink": sorted(streaming - in_memory)[:10],
        "only InMemorySink": sorted(in_memory - streaming)[:10],
    }
    assert not any(disagreement.values()), (
        f"{name}: the two backends annotate differently.\n"
        + "\n".join(f"  {side}: {entries}" for side, entries in disagreement.items())
    )
