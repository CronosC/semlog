"""`Reasoner`, and what a reasoner makes of each build of the matrix.

Most of these start a JVM. The ones that reject an argument or a JVM do not.
"""

from __future__ import annotations

from pathlib import Path

import jpype
import pytest

from semlog.core.enums import IndividualKind, OWLProfile, ReasonerBackend
from semlog.core.errors import ConfigError, DependencyError, InputError
from semlog.core.vocabulary import DEFAULT_IRI
from semlog.onto.reason import Reasoner, start_jvm
from semlog.onto.sinks import InMemorySink
from tests.support.rdf import read_rdf

# Pellet reads both profiles and accepts a timestamp with microseconds.
BACKEND = "Pellet"


def _small_ontology(path: Path) -> None:
    """Write an ontology with a class and a property assertion to entail from."""
    sink = InMemorySink(DEFAULT_IRI, OWLProfile.OWL2EL)
    sink.add_class("Human")
    sink.add_class("Man", ["Human"])
    sink.add_property("knows", domain_class=["Human"], range_class="Human")
    sink.assert_individuals("Man", ["a", "b"], IndividualKind.OBJECT)
    sink.assert_relations(
        "knows", [("a", "b")], (IndividualKind.OBJECT, IndividualKind.OBJECT)
    )
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
    reasoner = Reasoner(str(small))
    reasoner.materialize()
    reasoner.overwrite()

    assert read_rdf(small, DEFAULT_IRI).types["a"] == {"Man", "Human"}


@pytest.mark.reasoner
@pytest.mark.parametrize("backend", [b.value for b in ReasonerBackend])
def test_every_backend_materializes(backend, small):
    before = small.stat().st_size

    reasoner = Reasoner(str(small), backend=backend)
    reasoner.materialize()
    reasoner.overwrite()

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

    monkeypatch.setattr("semlog.onto.reason.SyncReasoner", fail)

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
        reasoner.overwrite()

    assert small.read_text(encoding="utf-8") == before
    assert [p.name for p in small.parent.iterdir()] == ["small.owl"]


@pytest.mark.reasoner
@pytest.mark.parametrize(
    ("suffix", "start"), [("ofn", "Prefix("), ("owl", "<?xml"), ("ttl", "@prefix")]
)
def test_a_save_writes_the_format_its_suffix_names(small, tmp_path, suffix, start):
    target = tmp_path / f"saved.{suffix}"

    Reasoner(str(small), backend=BACKEND).save(str(target))

    assert target.read_text(encoding="utf-8").startswith(start)


def _exhaust_the_heap(*args, **kwargs):
    raise jpype.JClass("java.lang.OutOfMemoryError")("Java heap space")


@pytest.mark.reasoner
def test_a_heap_exhaustion_while_loading_names_java_tool_options(small, monkeypatch):
    start_jvm()
    monkeypatch.setattr("semlog.onto.reason.SyncOntology", _exhaust_the_heap)

    with pytest.raises(DependencyError, match="JAVA_TOOL_OPTIONS=-Xmx") as failure:
        Reasoner(str(small))

    assert "Build the ontology again" not in str(failure.value)
    assert "\n" not in str(failure.value)


@pytest.mark.reasoner
def test_a_heap_exhaustion_wrapped_in_another_java_error_names_java_tool_options(
    small, monkeypatch
):
    reasoner = Reasoner(str(small), backend=BACKEND)

    def fail(*args, **kwargs):
        oom = jpype.JClass("java.lang.OutOfMemoryError")("Java heap space")
        raise jpype.JClass("java.lang.RuntimeException")("reasoning failed", oom)

    monkeypatch.setattr(reasoner.reasoner, "infer_axioms", fail)

    with pytest.raises(DependencyError, match="JAVA_TOOL_OPTIONS=-Xmx") as failure:
        reasoner.materialize()

    assert "Try one of the other backends" not in str(failure.value)


@pytest.mark.reasoner
def test_a_java_error_is_reported_on_one_line(small, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("first line\n\tat second line")

    monkeypatch.setattr("semlog.onto.reason.SyncReasoner", fail)

    with pytest.raises(InputError, match="first line at second line") as failure:
        Reasoner(str(small), backend="Pellet")

    assert "\n" not in str(failure.value)


def test_a_java_older_than_17_is_named_with_its_version(monkeypatch):
    monkeypatch.setattr(jpype, "isJVMStarted", lambda: True)
    monkeypatch.setattr(jpype, "getJVMVersion", lambda: (1, 8, 0))

    with pytest.raises(DependencyError, match=r"Java 17 or newer.*Java 1\.8\.0"):
        start_jvm()


def test_a_jvm_that_does_not_start_is_named_with_its_version(monkeypatch, tmp_path):
    home = tmp_path / "jdk"
    libjvm = home / "jre" / "lib" / "amd64" / "server" / "libjvm.so"
    libjvm.parent.mkdir(parents=True)
    libjvm.write_bytes(b"")
    (home / "release").write_text('JAVA_VERSION="1.8.0_292"\n', encoding="utf-8")

    def refuse(*args, **kwargs):
        raise jpype.JVMNotSupportedException("JVM mismatch, python is 64 bit")

    monkeypatch.setattr(jpype, "isJVMStarted", lambda: False)
    monkeypatch.setattr(jpype, "getDefaultJVMPath", lambda: str(libjvm))
    monkeypatch.setattr(jpype, "startJVM", refuse)

    with pytest.raises(DependencyError, match=r"Java 1\.8\.0_292 .* does not start"):
        start_jvm()


def test_a_heap_too_small_for_the_jvm_to_start_names_java_tool_options(monkeypatch):
    def exhaust(*args, **kwargs):
        raise SystemError("java.lang.OutOfMemoryError: Java heap space")

    monkeypatch.setattr(jpype, "isJVMStarted", lambda: False)
    monkeypatch.setattr(jpype, "startJVM", exhaust)

    with pytest.raises(DependencyError, match="JAVA_TOOL_OPTIONS=-Xmx"):
        start_jvm()
