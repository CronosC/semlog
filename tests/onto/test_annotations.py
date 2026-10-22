"""`AnnotationWriter` and the two annotation documents it fills."""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import InverseRepresentation
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.tables import EntityKind
from semanticlog.onto import annotations as annotations_module
from semanticlog.onto.annotations import (
    AnnotationWriter,
    _declare_label,
    derived_annotations,
    tbox_annotations,
    timestamp_labels,
)
from semanticlog.onto.settings import OntoBuildSettings
from tests.support.tbox import construct_tbox

# --------------------------------------------------------------------------
# The two documents
# --------------------------------------------------------------------------


def test_the_upper_schema_document_annotates_only_names_the_schema_declares():
    settings = OntoBuildSettings(
        inverse_representations=[
            InverseRepresentation.AXIOMATIC,
            InverseRepresentation.IMPLICIT,
        ]
    )
    onto = construct_tbox(DEFAULT_IRI, settings, attribute_history=True).require_onto()
    declared = {
        entity.iri.remainder
        for entity in [
            *onto.classes_in_signature(),
            *onto.object_properties_in_signature(),
            *onto.data_properties_in_signature(),
        ]
    }

    assert set(tbox_annotations()) - declared == set()


def test_every_derived_template_has_a_label_and_a_whole_inverse():
    for name, entry in derived_annotations().items():
        assert entry["label"], name
        assert entry.keys() <= {
            "label",
            "comment",
            "inverse_label",
            "inverse_comment",
        }, name
        assert ("inverse_label" in entry) == ("inverse_comment" in entry), name


def test_every_derived_template_is_both_in_the_document_and_written():
    source = Path(annotations_module.__file__).read_text(encoding="utf-8")
    named = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", "") not in ("write", "label_series"):
            continue
        first = node.args[0] if node.args else None
        assert isinstance(first, ast.Constant), (
            f"line {node.lineno}: the template has to be a string literal"
        )
        named.add(first.value)

    assert named == set(derived_annotations())


# --------------------------------------------------------------------------
# The text derived from the log
# --------------------------------------------------------------------------


@pytest.fixture
def writer(recording_sink):
    """Return a factory for a writer and the recording sink it writes to."""

    def make(
        inverses: bool = False, backmap: dict[str, str] | None = None, **overrides
    ):
        overrides.setdefault("derive_annotations", True)
        settings = OntoBuildSettings(
            inverse_representations=[InverseRepresentation.AXIOMATIC]
            if inverses
            else [],
            **overrides,
        )
        sink = recording_sink(settings.profile)
        return AnnotationWriter(sink, settings, backmap or {}), sink

    return make


def texts(sink) -> dict[tuple[str, str], str]:
    """Return every annotation written, keyed by name and label or comment."""
    return {(name, kind): text for _, name, kind, text in sink.of_kind("annotate")}


def test_nothing_is_written_without_derive_annotations(writer):
    annotations, sink = writer(derive_annotations=False)
    sink.add_class("place_order", source="a test")

    annotations.entity_class("place_order", EntityKind.EVENT)
    annotations.entity_individuals("place_order", ["e1"])

    assert sink.of_kind("annotate") == []


def test_an_entity_class_is_labeled_with_the_name_the_log_used(writer):
    annotations, sink = writer(backmap={"place_order": "place order"})
    for name in ("place_order", "order"):
        sink.add_class(name, source="a test")

    annotations.entity_class("place_order", EntityKind.EVENT)
    annotations.entity_class("order", EntityKind.OBJECT)

    assert texts(sink) == {
        ("place_order", "label"): "place order",
        ("place_order", "comment"): "Events of type 'place order'.",
        ("order", "label"): "order",
        ("order", "comment"): "Objects of type 'order'.",
    }


def test_an_entity_individual_is_labeled_with_its_type_and_its_original_id(writer):
    annotations, sink = writer(backmap={"place_order": "place order", "e_1": "e 1"})

    annotations.entity_individuals("place_order", ["e_1", None, "e2"])

    assert texts(sink) == {
        ("e_1", "label"): "place order (e 1)",
        ("e2", "label"): "place order (e2)",
    }


def test_the_order_super_properties_are_labeled_by_direction(writer):
    annotations, sink = writer(inverses=True)
    for prop in (vocab.DF, vocab.EF):
        sink.add_property(prop, create_inverse=True, source="a test")

    annotations.order_superproperties(include_df=True, include_ef=True)

    written = texts(sink)
    assert written[(vocab.DF, "label")] == "directly followed by"
    assert written[(vocab.inverse(vocab.DF), "label")] == "directly preceded by"
    assert written[(vocab.EF, "label")] == "eventually followed by"
    assert written[(vocab.inverse(vocab.EF), "label")] == "eventually preceded by"


def test_only_the_order_super_property_a_build_emits_is_annotated(writer):
    annotations, sink = writer()
    for prop in (vocab.DF, vocab.EF):
        sink.add_property(prop, source="a test")

    annotations.order_superproperties(include_df=True, include_ef=False)

    assert {name for name, _ in texts(sink)} == {vocab.DF}


def test_the_trace_boundary_classes_name_their_case_concept(writer):
    annotations, sink = writer(backmap={"purchase_order": "purchase order"})
    start, end = vocab.start_class("purchase_order"), vocab.end_class("purchase_order")
    for name in (start, end):
        sink.add_class(name, source="a test")

    annotations.trace_boundary_classes("purchase_order", start, end)

    written = texts(sink)
    assert written[(start, "label")] == "Start event (purchase order)"
    assert written[(end, "label")] == "End event (purchase order)"


def test_a_case_order_property_and_its_inverse_name_their_case_concept(writer):
    annotations, sink = writer(
        inverses=True, backmap={"purchase_order": "purchase order"}
    )
    df, ef = vocab.df_property("purchase_order"), vocab.ef_property("purchase_order")
    for prop in (df, ef):
        sink.add_property(prop, create_inverse=True, source="a test")

    annotations.df_property("purchase_order", df)
    annotations.ef_property("purchase_order", ef)

    written = texts(sink)
    assert written[(df, "label")] == "directly followed by (purchase order)"
    assert (
        written[(vocab.inverse(df), "label")] == "directly preceded by (purchase order)"
    )
    assert written[(ef, "label")] == "eventually followed by (purchase order)"


def test_an_inverse_is_not_annotated_when_the_build_has_none(writer):
    annotations, sink = writer(inverses=False)
    df = vocab.df_property("order")
    sink.add_property(df, source="a test")

    annotations.df_property("order", df)

    assert {name for name, _ in texts(sink)} == {df}


def test_an_attribute_annotates_its_class_its_property_and_the_inverse(writer):
    annotations, sink = writer(
        inverses=True,
        backmap={"order_unit_price": "unit price", "purchase_order": "purchase order"},
    )
    sink.add_class("order_unit_price", source="a test")
    sink.add_property("has_order_unit_price", create_inverse=True, source="a test")

    annotations.attribute(
        "order_unit_price",
        "has_order_unit_price",
        {EntityKind.OBJECT: ["purchase_order"]},
    )

    assert texts(sink) == {
        ("order_unit_price", "label"): "unit price (purchase order)",
        ("order_unit_price", "comment"): (
            "Values of the 'unit price' attribute of purchase order objects."
        ),
        ("has_order_unit_price", "label"): "has unit price",
        ("has_order_unit_price", "comment"): (
            "Relates purchase order objects to their 'unit price' value."
        ),
        ("inv_has_order_unit_price", "label"): "is unit price of",
        ("inv_has_order_unit_price", "comment"): (
            "Inverse of has unit price. Relates a 'unit price' value to the "
            "purchase order objects carrying it."
        ),
    }


def test_an_attribute_of_events_and_objects_names_every_owner_in_one_comment(writer):
    annotations, sink = writer()
    sink.add_class("weight", source="a test")
    sink.add_property("has_weight", source="a test")

    annotations.attribute(
        "weight",
        "has_weight",
        {EntityKind.EVENT: ["pack", "ship"], EntityKind.OBJECT: ["item"]},
    )

    assert texts(sink)[("weight", "comment")] == (
        "Values of the 'weight' attribute of pack and ship events and item objects."
    )


@pytest.mark.parametrize(
    ("label", "bounds", "comment"),
    [
        ("very_low", (-np.inf, 3.5), "that are at most 3.5."),
        ("low", (3.5, 10.25), "that are greater than 3.5 and at most 10.25."),
        ("very_high", (10.25, np.inf), "that are greater than 10.25."),
    ],
)
def test_a_bin_class_is_annotated_with_the_values_it_holds(
    writer, label, bounds, comment
):
    annotations, sink = writer()
    sink.add_class("price_bin_x", source="a test")

    annotations.bin_class(
        "price_bin_x", "price", {EntityKind.OBJECT: ["order"]}, label, bounds
    )

    assert texts(sink) == {
        ("price_bin_x", "label"): f"price (order): {label.replace('_', ' ')}",
        ("price_bin_x", "comment"): (
            f"Values of the 'price' attribute of order objects {comment}"
        ),
    }


def test_an_attribute_value_is_labeled_as_the_log_wrote_it(writer):
    annotations, sink = writer(backmap={"by_hand": "by hand"})

    annotations.attribute_values(
        {
            "by hand": "by_hand",
            10.5: "v_10_5",
            pd.Timestamp(2024, 1, 1): "t",
            pd.Timestamp(2024, 1, 1, 0, 0, 0, 500000): "t_frac",
        }
    )

    assert texts(sink) == {
        ("by_hand", "label"): "by hand",
        ("v_10_5", "label"): "10.5",
        ("t", "label"): "2024-01-01 00:00:00",
        ("t_frac", "label"): "2024-01-01 00:00:00.500000",
    }


def test_a_relation_is_annotated_with_the_types_it_connects(writer):
    annotations, sink = writer(inverses=True, backmap={"is_part_of": "is part of"})
    sink.add_property("is_part_of", create_inverse=True, source="a test")

    annotations.relation("is_part_of", ["order", "item"], ["shipment"])

    written = texts(sink)
    assert written[("is_part_of", "label")] == "is part of"
    assert (
        written[("is_part_of", "comment")]
        == "Relation 'is part of' from order or item to shipment."
    )
    assert written[("inv_is_part_of", "label")] == "inverse of is part of"
    assert (
        written[("inv_is_part_of", "comment")]
        == "Inverse of 'is part of'. Relates shipment to order or item."
    )


def test_a_relation_with_no_types_reads_as_from_an_event_to_an_object(writer):
    annotations, sink = writer()
    sink.add_property("concerns", source="a test")

    annotations.relation("concerns", [], [])

    assert (
        texts(sink)[("concerns", "comment")]
        == "Relation 'concerns' from event to object."
    )


def test_a_mined_constraint_is_labeled_with_its_template_and_activities(writer):
    annotations, sink = writer(backmap={"place_order": "place order"})
    for name in ("DeclareConstraint", "Response__place_order__ship__"):
        sink.add_class(name, source="a test")

    annotations.declare_constraints(
        "DeclareConstraint", ["Response__place_order__ship__"]
    )

    written = texts(sink)
    assert written[("DeclareConstraint", "label")] == "DECLARE constraint"
    assert (
        written[("Response__place_order__ship__", "label")]
        == "Response (place order, ship)"
    )


@pytest.mark.parametrize(
    ("constraint", "known", "expected"),
    [
        (
            "Response__Create___Update__Ship__",
            ["Create___Update", "Ship"],
            "Response (Create___Update, Ship)",
        ),
        ("Response__a__b__", [], "Response (a, b)"),
        ("Existence1", ["a"], "Existence1"),
    ],
)
def test_a_constraint_label_splits_on_known_activities_first(
    constraint, known, expected
):
    """A cleaned activity name can hold the `__` separator itself."""
    assert _declare_label(constraint, lambda part: part, known) == expected


def test_the_attribute_change_schema_names_the_attribute(writer):
    annotations, sink = writer(inverses=True, backmap={"unit_price": "unit price"})
    change_class = vocab.attribute_change_class("unit_price")
    sink.add_class(change_class, source="a test")
    sink.add_property("changes_unit_price", create_inverse=True, source="a test")

    annotations.attribute_change_schema(
        "unit_price", change_class, "changes_unit_price"
    )

    written = texts(sink)
    assert written[(change_class, "label")] == "unit price change"
    assert written[("changes_unit_price", "label")] == "changes unit price to"
    assert written[("inv_changes_unit_price", "label")] == "is new unit price in change"


def test_a_change_record_label_pairs_each_record_with_its_own_row(writer):
    annotations, sink = writer(backmap={"unit_price": "unit price"})

    annotations.attribute_change_individuals(
        change_ids=pd.Series(["c1", "c2"]),
        attr="unit_price",
        obj_type="order",
        object_ids=pd.Series(["o1", "o2"], index=[7, 9]),
        timestamps=pd.Series(
            [datetime(2024, 1, 1, 9, 0), datetime(2024, 1, 2, 17, 30)], index=[7, 9]
        ),
    )

    assert texts(sink) == {
        ("c1", "label"): "unit price change for order (o1) at 2024-01-01 09:00:00",
        ("c2", "label"): "unit price change for order (o2) at 2024-01-02 17:30:00",
    }


def test_only_the_upper_schema_this_build_declared_is_annotated(writer):
    annotations, sink = writer()
    sink.add_class(vocab.EVENT, source="a test")
    sink.add_class("Object", source="a log class named like a TBox class")

    annotations.tbox()

    assert {name for name, _ in texts(sink)} == {"Event"}


def test_a_timestamp_label_keeps_a_fraction_only_where_there_is_one():
    values = pd.Series(
        pd.to_datetime(
            [
                "2024-01-01T10:00:00.100000Z",
                "2024-01-01T10:00:00.900000Z",
                "2024-01-01T10:00:00.000000Z",
            ],
            utc=True,
        )
    )

    assert list(timestamp_labels(values)) == [
        "2024-01-01 10:00:00.100000",
        "2024-01-01 10:00:00.900000",
        "2024-01-01 10:00:00",
    ]


@pytest.mark.parametrize(
    ("template", "fields", "match"),
    [
        ("order.nope", {}, "is not an annotation template"),
        ("order.df_case", {}, "which was not supplied"),
    ],
)
def test_an_unknown_template_or_a_missing_field_is_rejected(
    writer, template, fields, match
):
    """A template with a field left over would emit a label holding '{}'."""
    annotations, sink = writer()
    sink.add_property("df_order", source="a test")

    with pytest.raises(ValueError, match=match):
        annotations.write(template, "df_order", **fields)
