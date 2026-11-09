"""`AxiomSink`: the checks every sink runs, and the conformance of the two real sinks.

The base-class checks run through `RecordingSink`. The conformance tests run
over `InMemorySink` and `StreamingSink`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import IndividualKind, OWLProfile
from semanticlog.core.errors import ConfigError
from semanticlog.core.vocabulary import (
    DEFAULT_IRI,
    TBOX_IRI,
    NameCollisionError,
)
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.sinks import InMemorySink, StreamingSink
from tests.support import stores
from tests.support.build import BACKENDS, IN_MEMORY, STREAMING, emit
from tests.support.owlapi import ontology_annotations
from tests.support.settings import CONFIGURATIONS, NAMES

EVENT = IndividualKind.EVENT
OBJECT = IndividualKind.OBJECT

OWL_VERSION_INFO = "http://www.w3.org/2002/07/owl#versionInfo"

# --------------------------------------------------------------------------
# The two real sinks agree
# --------------------------------------------------------------------------

COMPARED = (
    "declared",
    "classes",
    "object_properties",
    "data_properties",
    "individuals",
    "relations",
    "data_assertions",
    "annotations",
    "inverses",
    "transitive",
    "disjoint",
    "functional",
    "inverse_functional",
    "defined",
)


@pytest.mark.parametrize("name", NAMES)
def test_both_sinks_emit_the_same_ontology(name, built):
    streamed = built(name, STREAMING)
    in_memory = built(name, IN_MEMORY)

    differences = {
        field: (
            sorted(getattr(streamed, field) - getattr(in_memory, field))[:5],
            sorted(getattr(in_memory, field) - getattr(streamed, field))[:5],
        )
        for field in COMPARED
        if getattr(streamed, field) != getattr(in_memory, field)
    }
    assert differences == {}, f"{name}: the sinks disagree"


@pytest.mark.reasoner
def test_both_sinks_annotate_the_ontology_itself_with_the_same_record(built_path):
    streamed = ontology_annotations(built_path("dl_df", STREAMING))
    in_memory = ontology_annotations(built_path("dl_df", IN_MEMORY))

    assert streamed == in_memory
    assert len(in_memory) > 0


@pytest.mark.slow
def test_both_sinks_count_the_same_entities_and_axioms(tmp_path):
    for name in NAMES:
        config = CONFIGURATIONS[name]()
        streamed = StreamingSink(
            str(tmp_path / f"{name}.ofn"), DEFAULT_IRI, config.settings.profile
        )
        in_memory = InMemorySink(DEFAULT_IRI, config.settings.profile)
        for sink in (streamed, in_memory):
            OCEDOntoBuilder(DEFAULT_IRI, config.settings, sink=sink).build_ontology(
                stores.build(stores.MATRIX_STORE), CONFIGURATIONS[name]().inputs
            )

        assert streamed.declared_by_kind() == in_memory.declared_by_kind(), name
        assert streamed.axiom_count == in_memory.axiom_count, name


def test_a_class_with_a_parent_and_two_members_counts_four_axioms(real_sink):
    sink = real_sink()

    sink.add_class("order", "Object", source="a test")
    sink.assert_individuals("order", ["o1", "o2"], OBJECT)

    assert sink.axiom_count == 4


def test_text_with_quotes_backslashes_and_newlines_survives_unchanged(
    real_sink, read_back
):
    text = 'he said "hi"\nthen\ta\\b and a\\nb'
    sink = real_sink()
    sink.add_class("order", source="a test")
    sink.add_property("has_note", data_prop=True, source="a test")
    sink.assert_individuals("order", ["o1"], OBJECT)
    sink.assert_data("has_note", [("o1", text)], OBJECT)
    sink.annotate_entity("order", comment=text)

    onto = read_back(sink)

    assert onto.data_values == {("has_note", "o1", text, "string")}
    assert onto.annotations == {("order", "comment", text)}


def test_under_el_a_boolean_and_a_date_are_written_as_strings(real_sink, read_back):
    sink = real_sink(OWLProfile.OWL2EL)
    sink.add_class("order", source="a test")
    sink.add_property("has_flag", data_prop=True, source="a test")
    sink.assert_individuals("order", ["o1", "o2"], OBJECT)
    sink.assert_data("has_flag", [("o1", True), ("o2", date(2024, 1, 2))], OBJECT)

    assert read_back(sink).data_values == {
        ("has_flag", "o1", "true", "string"),
        ("has_flag", "o2", "2024-01-02", "string"),
    }


def test_both_sinks_write_a_float_as_a_decimal(tmp_path):
    streamed = tmp_path / "streamed.ofn"
    in_memory = tmp_path / "memory.owl"
    emit(CONFIGURATIONS["dl_df"](), streamed, STREAMING)
    emit(CONFIGURATIONS["dl_df"](), in_memory, IN_MEMORY)

    streamed_text = streamed.read_text(encoding="utf-8")
    written = in_memory.read_text(encoding="utf-8")
    assert "^^xsd:decimal" in streamed_text
    assert "^^xsd:double" not in streamed_text
    assert "XMLSchema#decimal" in written
    assert "XMLSchema#double" not in written


# --------------------------------------------------------------------------
# The base IRI
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "iri", ["", "http://example.org/a b#", "http://x>#", 'http://x.org/a"b/']
)
def test_a_base_iri_that_cannot_be_written_is_rejected(iri):
    with pytest.raises(ConfigError, match="^iri: .*cannot be written"):
        InMemorySink(iri, OWLProfile.OWL2DL)


def test_a_base_iri_without_a_slash_is_given_one(recording_sink):
    assert recording_sink(iri="https://example.org/upper").iri == (
        "https://example.org/upper/"
    )


SCHEMA = "https://example.com/schema#"


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_term_is_written_under_the_configured_tbox_iri(backend, tmp_path, read_back):
    sink = (
        StreamingSink(str(tmp_path / "s.ofn"), DEFAULT_IRI, OWLProfile.OWL2DL, SCHEMA)
        if backend == STREAMING
        else InMemorySink(DEFAULT_IRI, OWLProfile.OWL2DL, SCHEMA)
    )
    sink.add_class(vocab.EVENT, source="the TBox")
    sink.add_class("order", vocab.EVENT, source="a test")

    onto = read_back(sink)

    assert sink.iri_of(vocab.EVENT) == f"{SCHEMA}Event"
    assert onto.parents["order"] == {"Event"}
    if backend == STREAMING:
        text = (tmp_path / "s.ofn").read_text(encoding="utf-8")
        assert f"Prefix(oced:=<{SCHEMA}>)" in text
        assert TBOX_IRI not in text


def test_an_iri_ending_in_a_hash_is_rejected():
    with pytest.raises(ConfigError, match="^iri: must end in '/'"):
        InMemorySink("https://example.org/upper#", OWLProfile.OWL2DL)


# --------------------------------------------------------------------------
# What is written once and what is written every time
# --------------------------------------------------------------------------


def test_a_repeated_schema_statement_is_written_once(sink):
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

    assert {kind: len(sink.of_kind(kind)) for kind in {c[0] for c in sink.calls}} == {
        "declare": 2,
        "sub_class": 1,
        "sub_property": 1,
        "domain": 1,
        "range": 1,
        "transitive": 1,
        "annotate": 2,
    }


def test_an_iri_valued_annotation_is_written_once_on_both_sinks(real_sink, read_back):
    sink = real_sink()
    sink.add_class("pay", source="an activity")
    sink.add_class("Response__pay__", source="a constraint")
    sink.declare_annotation_property(vocab.ACTIVATION, "a test")
    sink.declare_annotation_property(vocab.TEMPLATE, "a test")
    for _ in range(2):
        sink.annotate_entity_with("Response__pay__", vocab.ACTIVATION, "pay", iri=True)
        sink.annotate_entity_with("Response__pay__", vocab.TEMPLATE, "Response")

    onto = read_back(sink)

    assert {a for a in onto.annotations if a[0] == "Response__pay__"} == {
        ("Response__pay__", "activation", "pay"),
        ("Response__pay__", "template", "Response"),
    }
    assert sink.axiom_count == 6


def test_an_annotation_naming_an_undeclared_entity_is_rejected(sink):
    sink.add_class("Response__pay__", source="a constraint")
    sink.declare_annotation_property(vocab.ACTIVATION, "a test")

    with pytest.raises(ValueError, match="'pay' is not declared"):
        sink.annotate_entity_with("Response__pay__", vocab.ACTIVATION, "pay", iri=True)


def test_a_row_level_assertion_is_written_every_time(sink):
    for _ in range(3):
        sink.assert_relations("df_order", [("e1", "e2")], (EVENT, EVENT))
        sink.assert_individuals("place_order", ["e1"], EVENT)

    assert len(sink.of_kind("relation")) == 3
    assert len(sink.of_kind("individual")) == 3


def test_a_class_under_two_parents_in_turn_keeps_both(sink):
    sink.add_class("channel", "ObjectAttribute")
    sink.add_class("channel", "EventAttribute")

    assert len(sink.of_kind("declare")) == 1
    assert {tuple(c[2]) for c in sink.of_kind("sub_class")} == {
        ("ObjectAttribute",),
        ("EventAttribute",),
    }


def test_a_class_under_several_parents_at_once_sits_under_their_union(sink):
    sink.add_class("channel", ["ObjectAttribute", "EventAttribute"])

    assert sink.of_kind("sub_class") == [
        ("sub_class", "channel", ["ObjectAttribute", "EventAttribute"])
    ]


def test_a_union_under_el_is_dropped_and_counted_once(recording_sink):
    sink = recording_sink(OWLProfile.OWL2EL)

    for _ in range(3):
        sink.add_property(
            "has_weight", domain_class=["Event", "Object"], range_class="weight"
        )

    assert sink.of_kind("domain") == []
    assert sink.dropped_unions == 1


def test_a_union_with_a_fallback_writes_the_fallback_under_el(recording_sink):
    sink = recording_sink(OWLProfile.OWL2EL)

    sink.add_property(
        "ships",
        domain_class=["pay", "ship"],
        range_class=["item", "order"],
        domain_fallback="Event",
        range_fallback="Object",
    )

    assert sink.of_kind("domain") == [("domain", "ships", ["Event"], False)]
    assert sink.of_kind("range") == [("range", "ships", ["Object"], False)]
    assert (sink.dropped_unions, sink.replaced_unions) == (0, 2)


def test_an_infinite_bin_edge_omits_its_facet(real_sink, tmp_path):
    sink = real_sink()
    sink.add_class("price", source="a test")
    sink.add_class("price_bin_low", "price", source="a test")
    sink.add_property("price_value", data_prop=True, source="a test")

    sink.add_bin_definition("price_bin_low", "price", "price_value", None, 6.5)
    sink.close()

    if isinstance(sink, StreamingSink):
        text = Path(sink.path).read_text(encoding="utf-8")
        assert "minExclusive" not in text
        assert 'xsd:maxInclusive "6.5"^^xsd:decimal' in text


def test_a_bin_definition_under_el_is_rejected(recording_sink):
    sink = recording_sink(OWLProfile.OWL2EL)

    with pytest.raises(ValueError, match="cannot be defined under OWL 2 EL"):
        sink.add_bin_definition("b", "a", "a_value", 1.0, None)


def test_a_functional_object_property_is_dropped_under_el(recording_sink):
    sink = recording_sink(OWLProfile.OWL2EL)
    sink.add_property("has_x", source="a test")

    sink.add_functional("has_x")
    sink.add_inverse_functional("has_x")

    assert sink.of_kind("functional") == []
    assert sink.of_kind("inverse_functional") == []
    assert sink.dropped_characteristics == 2


def test_a_functional_data_property_is_kept_under_el(recording_sink):
    sink = recording_sink(OWLProfile.OWL2EL)
    sink.add_property("value", data_prop=True, source="a test")

    sink.add_functional("value", data_prop=True)

    assert sink.of_kind("functional") == [("functional", "value", True)]
    assert sink.dropped_characteristics == 0


def test_a_functional_property_is_written_once_on_both_sinks(real_sink, read_back):
    sink = real_sink()
    sink.add_property("has_x", source="a test")
    sink.add_property("of_x", source="a test")

    for _ in range(2):
        sink.add_functional("has_x")
        sink.add_inverse_functional("of_x")

    emitted = read_back(sink)
    assert (emitted.functional, emitted.inverse_functional) == ({"has_x"}, {"of_x"})


def test_a_disjoint_set_is_written_once_on_both_sinks(real_sink, read_back):
    sink = real_sink()
    for name in ("a", "b", "c"):
        sink.add_class(name, source="a test")

    sink.add_disjoint_classes(["a", "b", "c"])
    sink.add_disjoint_classes(["c", "b", "a"])

    assert read_back(sink).disjoint == {frozenset({"a", "b", "c"})}


def test_a_disjoint_set_of_one_class_writes_nothing(sink):
    sink.add_class("a", source="a test")

    sink.add_disjoint_classes(["a", "a"])

    assert sink.of_kind("disjoint") == []


def test_a_union_with_a_fallback_is_kept_under_dl(sink):
    sink.add_property("ships", domain_class=["pay", "ship"], domain_fallback="Event")

    assert sink.of_kind("domain") == [("domain", "ships", ["pay", "ship"], False)]


@pytest.mark.parametrize("annotation", ["label", "comment"])
def test_a_second_annotation_text_on_one_entity_is_rejected(sink, annotation):
    sink.add_class("order", source="an object type")
    sink.annotate_entity("order", **{annotation: "derived text"})

    with pytest.raises(ValueError, match=f"'order' already has an rdfs:{annotation}"):
        sink.annotate_entity("order", **{annotation: "text from the document"})


def test_an_entity_annotated_twice_with_one_text_carries_one_label(
    real_sink, read_back
):
    sink = real_sink()
    sink.add_class("order", source="an object type")
    sink.annotate_entity("order", label="Order", comment="An order.")
    sink.annotate_entity("order", label="Order")

    assert read_back(sink).annotations == {
        ("order", "label", "Order"),
        ("order", "comment", "An order."),
    }


def test_a_control_character_becomes_a_replacement_character_on_both_sinks(
    real_sink, read_back
):
    sink = real_sink()
    sink.add_class("order", source="a test")
    sink.add_property("has_note", data_prop=True, source="a test")
    sink.assert_individuals("order", ["o1"], OBJECT)
    sink.assert_data("has_note", [("o1", "bad\x01char\t")], OBJECT)
    sink.annotate_entity("order", label="Or\x02der", comment="An\x1f order.")
    sink.annotate_individuals_bulk([("o1", "order\x0b(o1)")], OBJECT)

    onto = read_back(sink)

    assert onto.data_values == {("has_note", "o1", "bad\ufffdchar\t", "string")}
    assert onto.annotations == {
        ("order", "label", "Or\ufffdder"),
        ("order", "comment", "An\ufffd order."),
        ("o1", "label", "order\ufffd(o1)"),
    }


def test_the_values_with_control_characters_are_counted_by_property(sink):
    sink.add_class("order", source="a test")
    sink.add_property("has_note", data_prop=True, source="a test")
    sink.assert_individuals("order", ["o1", "o2", "o3"], OBJECT)
    sink.assert_data(
        "has_note", [("o1", "a\x01\x02"), ("o2", "b\x03"), ("o3", "clean")], OBJECT
    )
    sink.annotate_entity("order", label="Or\x02der")

    assert sink.replaced_controls == {"has_note": 2, "rdfs:label": 1}


def test_annotating_an_undeclared_entity_is_rejected(sink):
    with pytest.raises(ValueError, match="no build step declared it"):
        sink.annotate_entity("AttributeChange", label="Object attribute change")


def test_a_missing_value_is_not_asserted_as_an_individual(sink):
    sink.assert_individuals(
        "place_order", ["e1", None, float("nan"), pd.NaT, "e2"], EVENT
    )

    assert [call[1] for call in sink.of_kind("individual")] == ["e1", "e2"]


# --------------------------------------------------------------------------
# Names checked on the way through
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind", list(IndividualKind), ids=[k.value for k in IndividualKind]
)
def test_an_individual_is_written_under_its_kind_path(real_sink, read_back, kind):
    sink = real_sink()
    sink.add_class("thing", source="a test")
    sink.assert_individuals("thing", ["x1"], kind)

    onto = read_back(sink)

    assert onto.types["x1"] == {"thing"}
    if isinstance(sink, StreamingSink):
        assert f"ClassAssertion(:thing {kind.value}:x1)" in onto.axiom_lines


def test_a_class_and_an_event_of_one_name_are_two_entities(real_sink, read_back):
    sink = real_sink()
    sink.add_class("order", source="an object type")
    sink.add_class("place_order", source="an activity")
    sink.assert_individuals("place_order", ["order"], EVENT)
    sink.assert_relations("concerns", [("order", "order")], (EVENT, OBJECT))
    sink.annotate_individuals_bulk([("order", "the event")], EVENT)

    onto = read_back(sink)

    assert "order" in onto.classes
    assert onto.types["order"] == {"place_order"}
    assert ("concerns", "order", "order") in onto.relations


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"range_class": ["double", "integer"], "data_prop": True}, "is a list"),
        ({"range_class": "order", "data_prop": True}, "is not a datatype"),
        ({"range_class": "double"}, "is a datatype"),
        ({"data_prop": True, "create_inverse": True}, "cannot have an inverse"),
        ({"inverse_name": "p_inv"}, "inverse_name='p_inv' without create_inverse"),
        ({"axiomatic_inverse": True}, "axiomatic_inverse=True without"),
    ],
)
def test_property_arguments_that_contradict_each_other_are_rejected(
    sink, kwargs, match
):
    with pytest.raises(ValueError, match=match):
        sink.add_property("p", **kwargs)


def test_a_data_property_range_is_written_as_its_datatype(sink):
    sink.add_property("has_score", range_class="double", data_prop=True)

    assert sink.of_kind("range") == [("range", "has_score", ["double"], True)]


def test_the_inverse_of_a_parent_without_one_is_declared_before_use(sink):
    sink.add_property("relates", source="the TBox")
    sink.add_property("concerns", superprop="relates", create_inverse=True)

    declared = [call[1] for call in sink.of_kind("declare")]
    assert declared == ["relates", "concerns", "inv_concerns", "inv_relates"]
    assert ("sub_property", "inv_concerns", "inv_relates", False) in sink.calls


# --------------------------------------------------------------------------
# Opening and closing
# --------------------------------------------------------------------------


def test_the_record_declares_the_annotation_properties_in_the_tbox_namespace(sink):
    """OWL 2 DL requires a declaration for an annotation property."""
    sink.open_ontology(
        {f"{TBOX_IRI}buildTool": "semlog 0.5.0", OWL_VERSION_INFO: "semlog 0.5.0"}
    )

    assert [call[1] for call in sink.of_kind("declare")] == ["buildTool"]
    assert len(sink.of_kind("ontology_annotation")) == 2


def test_a_record_key_claims_its_name_as_an_annotation_property(sink):
    sink.open_ontology({f"{TBOX_IRI}buildTool": "semlog 0.5.0"})

    with pytest.raises(NameCollisionError, match="as annotation from the build record"):
        sink.add_class(vocab.BUILD_TOOL, source="an object type")


def test_a_log_class_named_like_a_schema_class_is_a_second_entity(sink):
    sink.add_class(vocab.EVENT, source="the TBox")
    sink.add_class("Event", source="an activity")

    declared = sink.of_kind("declare")
    assert [call[1] for call in declared] == ["Event", "Event"]
    assert [vocab.is_tbox(call[1]) for call in declared] == [True, False]


def test_the_record_after_the_first_axiom_is_rejected(sink):
    sink.add_class("order", source="a build step")

    with pytest.raises(ValueError, match="is open"):
        sink.open_ontology({f"{TBOX_IRI}buildTool": "semlog 0.5.0"})


def test_a_closed_sink_takes_no_more_axioms(sink):
    sink.close()

    with pytest.raises(ValueError, match="is closed"):
        sink.add_class("order", source="a build step")
