"""What a build says about the log it was built from.

Everything here reads the matrix store, whose contents are in `tests/support/stores.py`.
"""

from __future__ import annotations

import pytest

from semanticlog.core import vocabulary as vocab

EVENTS = [f"e{i}" for i in range(1, 13)]
SHIPMENTS = [f"s{i}" for i in range(1, 7)]
ORDERS = [f"o{i}" for i in range(1, 13)]
ITEMS = ["i1", "i2", "i3", "i4"]

VALUES_AND_EQUAL_BINS = "el_ef_explicit_bins"
EQUAL_BINS_ONLY = "el_history_bins_only"
DIST_BINS = "dl_ef_dist_bins_history"


def _bins(onto, attribute: str) -> dict[str, set[float]]:
    """Return the numbers under each bin class of one numeric attribute."""
    prefix = f"{attribute}_bin_"
    bins: dict[str, set[float]] = {}
    for name, classes in onto.types.items():
        for cls in classes:
            if cls.startswith(prefix):
                bins.setdefault(cls, set()).add(
                    float(name.removeprefix(f"{attribute}."))
                )
    return bins


def _pairs(onto, prop: str) -> set[tuple[str, str]]:
    return {(s, o) for p, s, o in onto.relations if p == prop}


@pytest.fixture(scope="module")
def onto(request):
    return request.getfixturevalue("built")("dl_df")


def test_each_type_of_the_log_is_a_subclass_of_its_schema_class(onto):
    assert onto.parents["place_order"] == {vocab.EVENT}
    assert onto.parents["order"] == {vocab.OBJECT}
    assert onto.parents["order.channel"] == {vocab.OBJECT_ATTRIBUTE}
    assert onto.parents["place_order.clerk"] == {vocab.EVENT_ATTRIBUTE}


def test_every_event_and_object_becomes_an_individual_of_its_type(onto):
    assert {e for e, t in onto.types.items() if "place_order" in t} == set(EVENTS)
    assert {e for e, t in onto.types.items() if "ship" in t} == set(SHIPMENTS)
    assert {o for o, t in onto.types.items() if "order" in t} == set(ORDERS)
    assert {i for i, t in onto.types.items() if "item" in t} == set(ITEMS)


def test_a_relation_of_the_log_becomes_an_assertion(onto):
    concerns = _pairs(onto, "concerns")

    assert len(concerns) == 18
    assert {("e1", "o1"), ("s1", "o1")} <= concerns
    assert _pairs(onto, "contains") == {
        ("o1", "i1"),
        ("o2", "i2"),
        ("o3", "i3"),
        ("o4", "i4"),
    }


def test_two_events_at_one_instant_share_one_time_individual(onto):
    times = {name for name, classes in onto.types.items() if vocab.TIME in classes}
    observed = _pairs(onto, "observed_at")

    # Eleven distinct stamps across twelve placements, plus six shipments.
    assert len(times) == 17
    assert len(observed) == 18
    assert len({o for s, o in observed if s in ("e11", "e12")}) == 1


def test_a_value_individual_is_shared_by_every_entity_of_its_type_carrying_it(onto):
    web = {s for s, o in _pairs(onto, "has_order.channel") if o == "order.channel.web"}

    assert {"o1", "o4"} <= web
    assert ("i1", "item.channel.web") in _pairs(onto, "has_item.channel")


def test_an_attribute_carried_by_two_types_gets_a_property_per_type(onto):
    assert onto.domains["has_order.channel"] == {"order"}
    assert onto.domains["has_item.channel"] == {"item"}
    assert "has_channel" not in onto.properties
    assert onto.domains["has_order.price"] == {"order"}


def test_a_missing_value_produces_no_assertion(onto):
    # o12 has no channel and no price.
    owners = {
        s for p, s, _ in onto.relations if p in ("has_order.channel", "has_order.price")
    }

    assert "o1" in owners
    assert "o12" not in owners


def test_directly_follows_links_consecutive_events_of_one_case(onto):
    # Every order has one placement, and the first six also have a shipment.
    assert _pairs(onto, "df_order") == {(f"e{i}", f"s{i}") for i in range(1, 7)}


def test_the_first_and_last_event_of_a_case_are_typed(onto):
    starts = {e for e, t in onto.types.items() if "StartEvent_order" in t}
    ends = {e for e, t in onto.types.items() if "EndEvent_order" in t}

    assert starts == set(EVENTS)
    assert ends == set(SHIPMENTS) | {f"e{i}" for i in range(7, 13)}


def test_a_name_cleaning_rewrote_is_emitted_in_its_cleaned_form(onto):
    assert "place_order" in onto.classes
    assert "place order" not in onto.classes
    assert "order.channel.by_hand" in onto.individuals


def test_the_build_records_where_it_came_from(onto):
    assert onto.ontology_annotations["sourceLog"] == "matrix"
    assert onto.ontology_annotations["versionInfo"].startswith("semlog ")
    assert '"profile": "OWL2DL"' in onto.ontology_annotations["buildConfiguration"]


# --------------------------------------------------------------------------
# Numeric attributes
# --------------------------------------------------------------------------


def test_values_keeps_one_individual_per_distinct_number(onto):
    prices = {name for name, t in onto.types.items() if "order.price" in t}
    literals = {v for p, _, v, _t in onto.data_values if p == vocab.HAS_DATA_VALUE}

    assert len(prices) == 11
    assert "order.price.10" in prices
    assert "10.0" in literals


def test_equal_bins_split_the_values_by_quantile(built):
    bins = _bins(built(EQUAL_BINS_ONLY), "order.price")

    # Eleven prices into two bins, so six fall below the median and five above.
    assert bins == {
        "order.price_bin_low": {3.25, 7.5, 10.0, 12.75, 18.0, 21.5},
        "order.price_bin_high": {25.5, 30.0, 33.25, 41.0, 120.0},
    }


def test_a_bin_class_sits_under_its_attribute(built):
    assert built(EQUAL_BINS_ONLY).parents["order.price_bin_low"] == {"order.price"}


def test_distribution_bins_follow_the_shape_of_the_column(built):
    assert _bins(built(DIST_BINS), "order.price") == {
        "order.price_bin_very_low": {3.25},
        "order.price_bin_low": {7.5, 10.0, 12.75},
        "order.price_bin_average": {18.0, 21.5, 25.5, 30.0},
        "order.price_bin_high": {33.25, 41.0},
        "order.price_bin_very_high": {120.0},
    }


def test_every_bin_class_carries_one_label_and_one_comment(built):
    onto = built(VALUES_AND_EQUAL_BINS)
    bin_classes = {c for c in onto.classes if "_bin_" in c}
    annotated = [(n, kind) for n, kind, _ in onto.annotations if n in bin_classes]

    assert bin_classes
    assert sorted(annotated) == sorted(
        (c, kind) for c in bin_classes for kind in ("comment", "label")
    )


def test_a_binned_value_keeps_its_number_only_with_values(built):
    with_values = built(VALUES_AND_EQUAL_BINS)
    without = built(EQUAL_BINS_ONLY)

    def priced(onto):
        return {n: t for n, t in onto.types.items() if n.startswith("order.price.")}

    assert len(priced(with_values)) == 11
    assert priced(with_values) == priced(without)
    assert {
        s for _, s, _v, _t in with_values.data_values if s.startswith("order.price.")
    } == set(priced(with_values))
    assert not {
        s for _, s, _v, _t in without.data_values if s.startswith("order.price.")
    }


def test_a_changed_number_lands_in_the_bin_of_the_same_number_on_an_object(built):
    onto = built(EQUAL_BINS_ONLY)
    bins = {c for c in onto.classes if c.startswith("order.price_bin_")}

    # o1 and o2 change their price to 12.75 and 41.0, which other orders also carry.
    assert onto.types["order.price.12.75"] & bins == {"order.price_bin_low"}
    assert onto.types["order.price.41"] & bins == {"order.price_bin_high"}


def test_no_numeric_representation_leaves_the_numbers_out(built):
    onto = built("dl_declare_multi_cc")

    assert "order.price" not in onto.declared
    assert "order.channel" in onto.classes


@pytest.mark.parametrize("name", [DIST_BINS, VALUES_AND_EQUAL_BINS])
def test_a_column_with_too_few_values_to_bin_keeps_its_numbers(built, name):
    # weight has four distinct values, below what either mode needs.
    onto = built(name)

    assert not any(c.startswith("item.weight_bin_") for c in onto.classes)
    assert "item.weight" in onto.classes
    assert {s for _, s, _v, _t in onto.data_values if s.startswith("item.weight.")}


# --------------------------------------------------------------------------
# DECLARE
# --------------------------------------------------------------------------


def test_each_case_object_is_typed_by_the_constraints_of_its_case_concept(built):
    onto = built("dl_declare_multi_cc")

    assert {
        t for t, c in onto.types.items() if "Response__place_order__ship__" in c
    } == {"o1"}
    assert {t for t, c in onto.types.items() if "Existence__ship__" in c} == {"i1"}
    assert onto.parents["Response__place_order__ship__"] == {
        "DeclareConstraint",
        "order",
    }


def test_a_mined_constraint_is_labeled_with_what_it_says(built):
    onto = built("dl_declare_multi_cc")
    labels = {name: text for name, kind, text in onto.annotations if kind == "label"}

    assert labels["Response__place_order__ship__"] == "Response (place order, ship)"
    assert labels["DeclareConstraint"] == "DECLARE constraint"


# --------------------------------------------------------------------------
# Inverses
# --------------------------------------------------------------------------


def test_implicit_inverses_assert_the_reverse_of_every_relation(built):
    onto = built("dl_inverses")
    reversed_properties = {p for p, _, _ in onto.relations if p.startswith("inv_")}

    assert {"inv_concerns", "inv_contains", "inv_df_order"} <= reversed_properties
    for inverse in sorted(reversed_properties):
        forward = _pairs(onto, inverse.removeprefix("inv_"))
        assert _pairs(onto, inverse) == {(o, s) for s, o in forward}, inverse


def test_an_inverse_that_reads_as_a_word_of_its_own_keeps_that_name(built):
    onto = built("dl_inverses")

    assert _pairs(onto, "observes") == {(o, s) for s, o in _pairs(onto, "observed_at")}
    assert "inv_observed_at" not in onto.properties


def test_the_inverse_of_every_attribute_property_sits_under_attribute_of(built):
    onto = built("dl_inverses")
    reversed_attributes = {p for p in onto.properties if p.startswith("inv_has_")}

    assert reversed_attributes == {
        "inv_has_place_order.amount",
        "inv_has_item.channel",
        "inv_has_order.channel",
        "inv_has_place_order.clerk",
        "inv_has_ship.clerk",
        "inv_has_ship.due",
        "inv_has_order.price",
        "inv_has_item.weight",
    }
    assert {p: onto.property_parents[p] for p in reversed_attributes} == {
        p: {"attribute_of"} for p in reversed_attributes
    }


def test_axiomatic_inverses_relate_the_two_properties(built):
    assert ("concerns", "inv_concerns") in built("dl_inverses").inverses


def test_el_gets_the_inverse_assertions_but_not_the_axiom(built):
    onto = built(VALUES_AND_EQUAL_BINS)

    assert ("inv_concerns", "o1", "e1") in onto.relations
    assert onto.inverses == set()
