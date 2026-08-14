"""The one contract both sinks are held to.

`InMemorySink` and `StreamingSink` write the same ontology through different
backends, so the same build has to come out the same either way. The checks
that live in the base class are exercised through `RecordingSink`, which
records the calls instead of writing anything.
"""

from __future__ import annotations

import pandas as pd
import pytest

from semanticlog.core.vocabulary import DEFAULT_IRI, UnsafeLocalNameError
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import InMemorySink, StreamingSink
from tests.conftest import IN_MEMORY, STREAMING, emit
from tests.support import stores
from tests.support.ontology import read_functional, read_rdf
from tests.support.settings import CONFIGURATIONS, NAMES


@pytest.mark.parametrize("name", NAMES)
def test_both_sinks_emit_the_same_ontology(name, built):
    streamed = built(name, STREAMING)
    in_memory = built(name, IN_MEMORY)

    differences = {
        field: (
            sorted(getattr(streamed, field) - getattr(in_memory, field))[:5],
            sorted(getattr(in_memory, field) - getattr(streamed, field))[:5],
        )
        for field in (
            "classes",
            "object_properties",
            "data_properties",
            "individuals",
            "relations",
            "data_assertions",
            "annotations",
            "inverses",
            "transitive",
        )
        if getattr(streamed, field) != getattr(in_memory, field)
    }
    assert not differences, f"{name}: the sinks disagree.\n" + "\n".join(
        f"  {field}: only streaming={only_s}, only in-memory={only_m}"
        for field, (only_s, only_m) in differences.items()
    )


@pytest.mark.parametrize("name", NAMES)
def test_both_sinks_declare_the_same_entities(name, built):
    assert built(name, STREAMING).declared == built(name, IN_MEMORY).declared


@pytest.mark.parametrize("name", NAMES)
def test_neither_sink_declares_anything_twice(name, built):
    for backend in (STREAMING, IN_MEMORY):
        onto = built(name, backend)
        assert len(onto.classes | onto.properties) == len(onto.classes) + len(
            onto.properties
        ), f"{name}/{backend} declares a name as both a class and a property"


@pytest.mark.parametrize("name", NAMES)
def test_nothing_is_used_before_the_build_declares_it(name, recording_sink):
    settings = CONFIGURATIONS[name]()
    sink = recording_sink(settings)

    OCEDOntoBuilder(DEFAULT_IRI, settings, sink=sink).build_ontology(
        stores.build(stores.MATRIX_STORE)
    )

    violations = sink.declaration_order_violations()
    assert not violations, (
        f"{name}: {len(violations)} name(s) were asserted as an individual and "
        "only declared as a class or property afterwards, which the sink's name "
        "ledger cannot detect:\n"
        + "\n".join(f"  {n!r} asserted into {cls!r}" for n, cls in violations)
    )


def test_the_declaration_order_check_would_notice_a_violation(recording_sink):
    sink = recording_sink(CONFIGURATIONS["dl_df"]())

    sink.assert_individuals(["late"], "some_class")
    sink.add_class("late", source="a class declared too late")

    assert sink.declaration_order_violations() == [("late", "some_class")]


# --------------------------------------------------------------------------
# The checks the base class runs, whichever backend is behind it
# --------------------------------------------------------------------------


@pytest.fixture
def sink(recording_sink):
    return recording_sink(CONFIGURATIONS["dl_df"]())


def test_a_repeated_schema_axiom_is_written_once(sink):
    for _ in range(3):
        sink.add_class("channel", "ObjectAttribute")
        sink.add_property(
            "has_channel",
            domain_class=["order", "item"],
            range_class="channel",
            superprop="has_attribute",
        )
        sink.add_transitive("ef_order")
        sink.annotate_entity("channel", label="channel", comment="an attribute")

    assert len(sink.of_kind("declare")) == 2
    assert len(sink.of_kind("sub_class")) == 1
    assert len(sink.of_kind("sub_property")) == 1
    assert len(sink.of_kind("domain")) == 1
    assert len(sink.of_kind("range")) == 1
    assert len(sink.of_kind("transitive")) == 1
    assert len(sink.of_kind("annotate")) == 2


def test_a_row_level_assertion_is_never_deduplicated(sink):
    for _ in range(3):
        sink.write_object_assertions("df_order", [("e1", "e2")])
        sink.write_class_assertions("place_order", ["e1"])

    assert len(sink.of_kind("relation")) == 3
    assert len(sink.of_kind("individual")) == 3


def test_an_unsafe_declared_name_is_refused(sink):
    with pytest.raises(UnsafeLocalNameError, match="bad name"):
        sink.add_class("bad name")
    with pytest.raises(UnsafeLocalNameError, match="has:colon"):
        sink.add_property("has:colon")
    with pytest.raises(UnsafeLocalNameError):
        sink.add_transitive("ef order")


def test_a_plus_or_a_tilde_is_refused_as_a_name(sink):
    """Neither can appear in an XML element name, so RDF/XML cannot write it."""
    with pytest.raises(UnsafeLocalNameError, match=r"a\+b"):
        sink.add_class("a+b")
    with pytest.raises(UnsafeLocalNameError, match="has_a~b"):
        sink.add_property("has_a~b")


def test_a_hyphen_or_a_dot_is_accepted_in_a_name(sink):
    sink.add_class("web-workers")
    sink.add_property("has_unit.price")

    assert sink.is_declared("web-workers")
    assert sink.is_declared("has_unit.price")


def test_a_class_expression_is_a_list_of_names_and_not_a_grammar(sink):
    sink.add_class("channel", ["ObjectAttribute", "EventAttribute"])
    sink.add_property("has_channel", domain_class=["order", "item"])

    assert sink.of_kind("sub_class")[0][2] == ["ObjectAttribute", "EventAttribute"]
    assert sink.of_kind("domain")[0][2] == ["order", "item"]

    with pytest.raises(UnsafeLocalNameError, match="B and C"):
        sink.add_class("A", "B and C")


def test_a_single_parent_is_still_accepted_as_a_bare_string(sink):
    sink.add_class("channel", "ObjectAttribute")

    assert sink.of_kind("sub_class")[0][2] == ["ObjectAttribute"]


def test_a_class_under_two_parents_keeps_both(sink):
    sink.add_class("channel", "ObjectAttribute")
    sink.add_class("channel", "EventAttribute")

    assert len(sink.of_kind("declare")) == 1
    assert {tuple(c[2]) for c in sink.of_kind("sub_class")} == {
        ("ObjectAttribute",),
        ("EventAttribute",),
    }


def test_a_different_annotation_on_the_same_entity_is_kept(sink):
    sink.add_class("order", source="an object type")
    sink.annotate_entity("order", label="derived label")
    sink.annotate_entity("order", label="label from the annotation document")

    assert len(sink.of_kind("annotate")) == 2


def test_annotating_an_undeclared_entity_is_refused(sink):
    with pytest.raises(ValueError, match="no build step declared it"):
        sink.annotate_entity("AttributeChange", label="Object attribute change")


def test_a_missing_value_is_not_asserted_as_an_individual(sink):
    """It has no name, and every one of them would arrive under the same one."""
    sink.assert_individuals(["e1", None, float("nan"), pd.NaT, "e2"], "place_order")

    assert [call[1] for call in sink.of_kind("individual")] == ["e1", "e2"]


def test_a_column_of_nothing_but_missing_values_asserts_nothing(sink):
    sink.assert_individuals([None, None], "place_order")

    assert sink.of_kind("individual") == []


def test_individuals_are_deduplicated_by_the_name_they_are_emitted_under(sink):
    sink.assert_individuals([1, "1", 2], "place_order")

    assert [call[1] for call in sink.of_kind("individual")] == ["1", "2"]


# --------------------------------------------------------------------------
# What only the streaming sink has to get right
# --------------------------------------------------------------------------


def test_a_streamed_file_is_a_complete_ontology(tmp_path):
    path = tmp_path / "streamed.ofn"
    emit(CONFIGURATIONS["dl_df"](), path)

    text = path.read_text(encoding="utf-8")
    assert text.startswith("Prefix(")
    assert text.rstrip().endswith(")")


AWKWARD_TEXT = {
    "newline": "first\nsecond",
    "tab": "first\tsecond",
    "quote": 'he said "hello"',
    "backslash": "a\\b",
    "backslash_n": "a\\nb",
}


def _write_text(sink, text: str) -> None:
    """Put one awkward string through every place a sink writes text."""
    sink.add_class("order", source="a test")
    sink.add_property("has_note", data_prop=True, source="a test")
    sink.write_class_assertions("order", ["o1"])
    sink.write_data_assertions("has_note", [("o1", text)])
    sink.annotate_entity("order", comment=text)


@pytest.mark.parametrize("case", sorted(AWKWARD_TEXT))
def test_text_survives_both_sinks_unchanged(case, tmp_path):
    """A literal is data. Whatever the syntax needs, the text has to come back."""
    text = AWKWARD_TEXT[case]
    settings = OntoBuildSettings(case_concepts=["order"])

    streamed_path = tmp_path / f"{case}.ofn"
    streamed_sink = StreamingSink(str(streamed_path), DEFAULT_IRI, settings)
    _write_text(streamed_sink, text)
    streamed_sink.save(str(streamed_path))
    streamed = read_functional(streamed_path)

    rdf_path = tmp_path / f"{case}.owl"
    in_memory_sink = InMemorySink(DEFAULT_IRI, settings)
    _write_text(in_memory_sink, text)
    in_memory_sink.save(str(rdf_path))
    in_memory = read_rdf(rdf_path, DEFAULT_IRI)

    assert streamed.data_values == {("has_note", "o1", text, "string")}
    assert in_memory.data_values == {("has_note", "o1", text, "string")}
    assert streamed.annotations == {("order", "comment", text)}
    assert in_memory.annotations == {("order", "comment", text)}


def test_typed_literals_survive_streaming(tmp_path):
    path = tmp_path / "streamed.ofn"
    emit(CONFIGURATIONS["dl_df"](), path)

    text = path.read_text(encoding="utf-8")
    assert "^^xsd:dateTime" in text
    assert "^^xsd:double" in text
    assert "^^xsd:integer" in text


def test_the_sinks_write_a_float_under_different_datatypes(tmp_path):
    """owlready2 writes every float as xsd:decimal, whatever datatype the
    literal declares, and every other serialisation it offers is converted
    from that RDF/XML. tests/support/ontology.py reads the two as one so the
    sink comparison stays about everything else, and this is where the
    difference is recorded.
    """
    streamed = tmp_path / "streamed.ofn"
    in_memory = tmp_path / "memory.owl"
    emit(CONFIGURATIONS["dl_df"](), streamed, STREAMING)
    emit(CONFIGURATIONS["dl_df"](), in_memory, IN_MEMORY)

    written = in_memory.read_text(encoding="utf-8")
    assert "^^xsd:double" in streamed.read_text(encoding="utf-8")
    assert "XMLSchema#decimal" in written
    assert "XMLSchema#double" not in written


def test_a_streamed_build_reports_its_path_and_has_no_ontology(tmp_path):
    path = tmp_path / "streamed.ofn"
    settings = CONFIGURATIONS["dl_df"]()
    sink = StreamingSink(str(path), DEFAULT_IRI, settings)

    result = OCEDOntoBuilder(DEFAULT_IRI, settings, sink=sink).build_ontology(
        stores.build(stores.MATRIX_STORE)
    )

    assert result.path == str(path)
    assert result.onto is None
    with pytest.raises(ValueError, match="streamed"):
        result.require_onto()


def test_saving_a_streaming_sink_elsewhere_is_refused(tmp_path):
    sink = StreamingSink(
        str(tmp_path / "streamed.ofn"), DEFAULT_IRI, OntoBuildSettings()
    )

    with pytest.raises(ValueError, match="cannot save"):
        sink.save(str(tmp_path / "elsewhere.ofn"))


def test_aborting_a_streaming_sink_leaves_no_file(tmp_path):
    """A partial ontology that parses is worse than no ontology at all."""
    path = tmp_path / "streamed.ofn"
    sink = StreamingSink(str(path), DEFAULT_IRI, OntoBuildSettings())
    sink.add_class("order", source="a build step")

    sink.abort()

    assert not path.exists()


def test_aborting_an_in_memory_sink_only_closes_it(tmp_path):
    sink = InMemorySink(DEFAULT_IRI, OntoBuildSettings())
    sink.add_class("order", source="a build step")

    sink.abort()

    assert sink.declared_count == 1


# --------------------------------------------------------------------------
# What a build reports about itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_both_sinks_count_the_same_entities_and_axioms(name, tmp_path):
    """The statistics a run logs have to mean the same thing for either sink."""
    settings = CONFIGURATIONS[name]()
    streamed = StreamingSink(str(tmp_path / f"{name}.ofn"), DEFAULT_IRI, settings)
    in_memory = InMemorySink(DEFAULT_IRI, settings)
    for sink in (streamed, in_memory):
        OCEDOntoBuilder(DEFAULT_IRI, CONFIGURATIONS[name](), sink=sink).build_ontology(
            stores.build(stores.MATRIX_STORE)
        )

    assert streamed.declared_by_kind() == in_memory.declared_by_kind()
    assert streamed.axiom_count == in_memory.axiom_count


def test_declared_entities_are_counted_by_kind(tmp_path):
    sink = InMemorySink(DEFAULT_IRI, OntoBuildSettings())
    sink.add_class("order", source="a build step")
    sink.add_property("concerns", source="a build step")
    sink.add_property("has_data_value", data_prop=True, source="a build step")

    assert sink.declared_by_kind() == {
        "classes": 1,
        "object_properties": 1,
        "data_properties": 1,
    }


def test_declared_names_are_reported_from_a_mark_in_declaration_order():
    sink = InMemorySink(DEFAULT_IRI, OntoBuildSettings())
    sink.add_class("order", source="a build step")
    mark = sink.declared_count
    sink.add_class("item", source="a build step")
    sink.add_class("shipment", source="a build step")
    sink.add_property("concerns", source="a build step")
    sink.add_property("has_data_value", data_prop=True, source="a build step")

    assert sink.declared_since(mark) == {
        "class": ["item", "shipment"],
        "object": ["concerns"],
        "data": ["has_data_value"],
    }


def test_declared_names_are_empty_for_a_step_that_declared_nothing():
    sink = InMemorySink(DEFAULT_IRI, OntoBuildSettings())
    sink.add_class("order", source="a build step")

    assert sink.declared_since(sink.declared_count) == {}


def test_the_in_memory_sink_keeps_the_assertions_implicit_inverses_materialize(built):
    """The backend links an inverse pair when the axiom arrives, and from then
    on keeps one direction of every assertion between the two properties. The
    sink holds the axioms back until the build is closed, so both survive.
    """
    onto = built("dl_inverses", IN_MEMORY)

    forward = {(s, o) for p, s, o in onto.relations if p == "concerns"}
    backward = {(s, o) for p, s, o in onto.relations if p == "inv_concerns"}

    assert forward
    assert backward == {(o, s) for s, o in forward}
    assert ("concerns", "inv_concerns") in onto.inverses


def test_the_inverse_axioms_are_written_even_without_a_save(tiny_store):
    from semanticlog.onto.settings import InverseRepresentation

    settings = OntoBuildSettings(
        inverse_representations=[
            InverseRepresentation.AXIOMATIC,
            InverseRepresentation.IMPLICIT,
        ]
    )
    semlog = OCEDOntoBuilder(DEFAULT_IRI, settings).build_ontology(tiny_store)

    axioms = str(list(semlog.require_onto().object_properties_in_signature()))
    assert "inv_concerns" in axioms
