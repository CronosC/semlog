"""`Reasoner`, and what a reasoner makes of each build of the matrix.

Most of these start a JVM. The two that reject an argument do not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from semanticlog.core.enums import OWLProfile
from semanticlog.core.errors import ConfigError, InputError
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.onto.reason import REASONER_BACKENDS, Reasoner
from semanticlog.onto.sinks import InMemorySink
from tests.support.rdf import read_rdf

# Pellet reads both profiles and accepts a timestamp with microseconds.
BACKEND = "Pellet"


def _small_ontology(path: Path) -> None:
    """Write an ontology with a class and a property assertion to entail from."""
    sink = InMemorySink(DEFAULT_IRI, OWLProfile.OWL2EL)
    sink.add_class("Human")
    sink.add_class("Man", ["Human"])
    sink.add_property("knows", domain_class=["Human"], range_class="Human")
    sink.assert_individuals("Man", ["a", "b"])
    sink.assert_relations("knows", [("a", "b")])
    sink.save(str(path))


@pytest.fixture
def small(tmp_path) -> Path:
    path = tmp_path / "small.owl"
    _small_ontology(path)
    return path


def test_an_unknown_backend_is_rejected(small):
    with pytest.raises(ConfigError, match="'Konclude' is not one of"):
        Reasoner(str(small), backend="Konclude")


def test_a_missing_ontology_file_is_rejected_before_the_jvm_starts(tmp_path):
    with pytest.raises(FileNotFoundError, match="Build it first with 'semlog build'"):
        Reasoner(str(tmp_path / "never-built.owl"))


@pytest.mark.reasoner
def test_materializing_writes_the_inferred_types_back_to_the_file(small):
    Reasoner(str(small)).materialize()

    assert read_rdf(small, DEFAULT_IRI).types["a"] == {"Man", "Human"}


@pytest.mark.reasoner
@pytest.mark.parametrize("backend", REASONER_BACKENDS)
def test_every_backend_materializes(backend, small):
    before = small.stat().st_size

    Reasoner(str(small), backend=backend).materialize()

    assert small.stat().st_size > before


@pytest.mark.reasoner
def test_materializing_returns_the_number_of_axioms_added(small):
    reasoner = Reasoner(str(small), backend=BACKEND)
    owlapi = reasoner.ontology.get_owlapi_ontology()
    before = owlapi.getAxiomCount()

    added = reasoner.materialize()

    assert added == owlapi.getAxiomCount() - before
    assert added > 0


@pytest.mark.reasoner
def test_a_file_the_owl_api_cannot_read_is_reported_as_the_file(tmp_path):
    path = tmp_path / "not-an-ontology.owl"
    path.write_text("this is not an ontology", encoding="utf-8")

    with pytest.raises(InputError, match="cannot read") as failure:
        Reasoner(str(path))

    assert "Try one of the other backends" not in str(failure.value)


@pytest.mark.reasoner
def test_a_backend_that_cannot_start_is_named_with_the_others(small, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("java.lang.IllegalStateException")

    monkeypatch.setattr("semanticlog.onto.reason.SyncReasoner", fail)

    with pytest.raises(InputError, match="Pellet cannot start") as failure:
        Reasoner(str(small), backend="Pellet")

    assert "HermiT" in str(failure.value)


@pytest.mark.reasoner
def test_a_backend_that_fails_mid_run_names_itself_and_the_step(small, monkeypatch):
    reasoner = Reasoner(str(small), backend=BACKEND)

    def fail(*args, **kwargs):
        raise RuntimeError("java.lang.UnsupportedOperationException")

    monkeypatch.setattr(reasoner.reasoner, "infer_axioms", fail)

    with pytest.raises(
        InputError, match=f"{BACKEND} failed while asking for entailments"
    ) as failure:
        reasoner.materialize()

    assert "Try one of the other backends" in str(failure.value)


@pytest.mark.reasoner
def test_a_save_that_fails_leaves_the_file_that_was_there(small, monkeypatch):
    """The OWL API opens the destination before it serializes anything."""
    before = small.read_text(encoding="utf-8")
    reasoner = Reasoner(str(small), backend=BACKEND)

    def fail(target, *args, **kwargs):
        Path(target).write_text("", encoding="utf-8")
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(reasoner.ontology, "save", fail)

    with pytest.raises(RuntimeError, match="serialization failed"):
        reasoner.save()

    assert small.read_text(encoding="utf-8") == before
    assert [p.name for p in small.parent.iterdir()] == ["small.owl"]
