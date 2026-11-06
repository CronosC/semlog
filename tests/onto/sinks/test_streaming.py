from __future__ import annotations

import pytest

from semanticlog.core.enums import IndividualKind, OWLProfile
from semanticlog.core.vocabulary import DEFAULT_IRI, EVENT, TBOX_IRI
from semanticlog.onto.sinks import StreamingSink

PROFILE = OWLProfile.OWL2DL


@pytest.fixture
def path(tmp_path):
    return tmp_path / "streamed.ofn"


@pytest.fixture
def streaming(path):
    return StreamingSink(str(path), DEFAULT_IRI, PROFILE)


def test_a_schema_name_is_written_under_the_tbox_prefix(streaming, path):
    streaming.add_class(EVENT, source="the TBox")
    streaming.add_class("order", EVENT, source="an object type")
    streaming.close()

    text = path.read_text(encoding="utf-8")
    assert f"Prefix(oced:=<{TBOX_IRI}>)" in text
    assert "Declaration(Class(oced:Event))" in text
    assert "SubClassOf(:order oced:Event)" in text


def test_the_header_declares_one_prefix_per_kind(streaming, path):
    streaming.close()

    text = path.read_text(encoding="utf-8")
    for kind in IndividualKind:
        assert f"Prefix({kind.value}:=<{DEFAULT_IRI}{kind.value}/>)" in text


def test_the_file_appears_only_once_the_ontology_is_complete(streaming, path):
    streaming.add_class("order", source="a build step")

    assert not path.exists()

    streaming.close()

    assert [p.name for p in path.parent.iterdir()] == ["streamed.ofn"]


def test_a_sink_given_no_record_writes_a_complete_ontology(streaming, path):
    streaming.add_class("order", source="a build step")

    streaming.close()

    text = path.read_text(encoding="utf-8")
    assert text.startswith("Prefix(")
    assert f"Ontology(<{DEFAULT_IRI.removesuffix('/')}>" in text
    assert text.rstrip().endswith(")")


def test_the_record_is_written_inside_the_ontology_header(streaming, path):
    streaming.open_ontology({TBOX_IRI + "buildTool": "semlog 0.5.0"})
    streaming.add_class("order", source="a build step")
    streaming.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines.index(f"Ontology(<{DEFAULT_IRI.removesuffix('/')}>")
    annotation = lines.index(f'Annotation(<{TBOX_IRI}buildTool> "semlog 0.5.0")')
    declaration = lines.index("Declaration(Class(:order))")
    assert header < annotation < declaration


def test_closing_twice_leaves_the_file_as_the_first_close_wrote_it(streaming, path):
    streaming.add_class("order", source="a build step")
    streaming.close()
    written = path.read_text(encoding="utf-8")

    streaming.close()

    assert path.read_text(encoding="utf-8") == written


def test_saving_to_another_path_is_rejected(streaming, tmp_path):
    with pytest.raises(ValueError, match="cannot save"):
        streaming.save(str(tmp_path / "elsewhere.ofn"))


def test_aborting_leaves_no_file(streaming, tmp_path):
    streaming.add_class("order", source="a build step")

    streaming.abort()

    assert list(tmp_path.iterdir()) == []


def test_leaving_the_context_writes_the_file(path):
    with StreamingSink(str(path), DEFAULT_IRI, PROFILE) as sink:
        sink.add_class("order", source="a build step")

    assert [p.name for p in path.parent.iterdir()] == ["streamed.ofn"]
    assert path.read_text(encoding="utf-8").rstrip().endswith(")")


def test_an_exception_inside_the_context_leaves_no_file(path):
    def failing_build() -> None:
        with StreamingSink(str(path), DEFAULT_IRI, PROFILE) as sink:
            sink.add_class("order", source="a build step")
            raise RuntimeError("build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        failing_build()

    assert list(path.parent.iterdir()) == []
