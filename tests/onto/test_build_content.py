"""What the emitted ontology says about the log it was built from.

The shape table records how much a configuration emits. These check that what
it emits means the right thing: that events and objects land under the right
schema class, that an attribute value is shared by everything carrying it,
that order and time follow the log, and that the provenance record is there.

Everything here reads the matrix store, whose contents are in
`tests/support/stores.py`.
"""

from __future__ import annotations

import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import (
    DataRepresentation,
    EventOrderRepresentation,
)
from tests.conftest import emit
from tests.support.settings import declare_constraints, settings

EVENTS = [f"e{i}" for i in range(1, 13)]
SHIPMENTS = [f"s{i}" for i in range(1, 7)]
ORDERS = [f"o{i}" for i in range(1, 13)]
ITEMS = ["i1", "i2", "i3", "i4"]


def _bins(onto, attribute: str) -> dict[str, set[float]]:
    """Return the numbers under each bin class of one numeric attribute."""
    prefix = f"{attribute}_bin_"
    bins: dict[str, set[float]] = {}
    for name, classes in onto.types.items():
        for cls in classes:
            if cls.startswith(prefix):
                value = float(name.removeprefix(f"{attribute}_"))
                bins.setdefault(cls, set()).add(value)
    return bins


@pytest.fixture(scope="module")
def onto(request):
    """The default configuration's build, shared by the module."""
    return request.getfixturevalue("built")("dl_df")


def test_an_activity_is_a_subclass_of_event(onto):
    assert onto.parents["place_order"] == {vocab.EVENT}
    assert onto.parents["ship"] == {vocab.EVENT}


def test_an_object_type_is_a_subclass_of_object(onto):
    assert onto.parents["order"] == {vocab.OBJECT}
    assert onto.parents["item"] == {vocab.OBJECT}


def test_an_attribute_is_a_subclass_of_the_attribute_class_of_its_owners(onto):
    assert onto.parents["channel"] == {vocab.OBJECT_ATTRIBUTE}
    assert onto.parents["clerk"] == {vocab.EVENT_ATTRIBUTE}


def test_every_event_and_object_becomes_an_individual_of_its_type(onto):
    assert {e for e, t in onto.types.items() if "place_order" in t} == set(EVENTS)
    assert {e for e, t in onto.types.items() if "ship" in t} == set(SHIPMENTS)
    assert {o for o, t in onto.types.items() if "order" in t} == set(ORDERS)
    assert {i for i, t in onto.types.items() if "item" in t} == set(ITEMS)


def test_a_relation_of_the_log_becomes_an_assertion(onto):
    concerns = {(s, o) for p, s, o in onto.relations if p == "concerns"}
    assert ("e1", "o1") in concerns
    assert ("s1", "o1") in concerns
    assert len(concerns) == 18

    contains = {(s, o) for p, s, o in onto.relations if p == "contains"}
    assert contains == {("o1", "i1"), ("o2", "i2"), ("o3", "i3"), ("o4", "i4")}


def test_two_events_at_one_instant_share_one_time_individual(onto):
    times = {name for name, classes in onto.types.items() if vocab.TIME in classes}
    observed = {(s, o) for p, s, o in onto.relations if p == "observed_at"}

    # Eleven distinct stamps across twelve placements, plus six shipments.
    assert len(times) == 17
    assert len(observed) == 18

    tie = {o for s, o in observed if s in ("e11", "e12")}
    assert len(tie) == 1, "e11 and e12 share a timestamp and so share a Time individual"


def test_a_value_individual_is_shared_by_everything_carrying_it(onto):
    has_channel = {(s, o) for p, s, o in onto.relations if p == "has_channel"}
    web = {s for s, o in has_channel if o == "channel_web"}

    assert {"o1", "o4", "i1"} <= web, (
        "channel_web is one individual, reached from every order and item on the web"
    )


def test_an_attribute_carried_by_two_types_gets_both_as_its_domain(onto):
    assert onto.domains["has_channel"] == {"order", "item"}
    assert onto.domains["has_price"] == {"order"}


def test_a_missing_value_produces_no_assertion(onto):
    # o12 has no channel and no price.
    owners = {s for p, s, _ in onto.relations if p in ("has_channel", "has_price")}
    assert "o12" not in owners


def test_directly_follows_links_consecutive_events_of_one_case(onto):
    df = {(s, o) for p, s, o in onto.relations if p == "df_order"}

    # Every order has one placement, and the first six also have a shipment.
    assert df == {(f"e{i}", f"s{i}") for i in range(1, 7)}


def test_the_first_and_last_event_of_a_case_are_typed(onto):
    starts = {e for e, t in onto.types.items() if "StartEvent_order" in t}
    ends = {e for e, t in onto.types.items() if "EndEvent_order" in t}

    assert starts == set(EVENTS)
    assert ends == set(SHIPMENTS) | {f"e{i}" for i in range(7, 13)}


def test_a_name_cleaning_rewrote_is_emitted_in_its_cleaned_form(onto):
    assert "place_order" in onto.classes
    assert "place order" not in onto.classes
    assert "channel_by_hand" in onto.individuals


def test_the_build_records_where_it_came_from(onto):
    assert onto.ontology_annotations["sourceLog"] == "matrix"
    assert onto.ontology_annotations["versionInfo"].startswith("semlog ")
    assert '"profile": "OWL2DL"' in onto.ontology_annotations["buildConfiguration"]


# --------------------------------------------------------------------------
# How numeric attributes are represented
# --------------------------------------------------------------------------


def test_values_keeps_one_individual_per_distinct_number(built):
    onto = built("dl_df")
    prices = {name for name, t in onto.types.items() if "price" in t}

    assert len(prices) == 11
    assert "price_10.0" in prices
    literals = {v for p, _, v, _t in onto.data_values if p == vocab.HAS_DATA_VALUE}
    assert "10.0" in literals


def test_bins_replace_the_numbers_with_labelled_classes(built):
    onto = built("dl_bins_only")

    assert {"price_bin_low", "price_bin_high"} <= onto.classes
    assert onto.parents["price_bin_low"] == {"price"}
    assert not {name for name, t in onto.types.items() if t == {"price"}}, (
        "with bins, every price individual is typed by a bin as well as by price"
    )


def test_equal_bins_split_the_values_by_quantile(built):
    onto = built("dl_bins_only")
    bins = _bins(onto, "price")

    # Eleven prices into two bins, so six fall below the median and five above.
    assert bins["price_bin_low"] == {3.25, 7.5, 10.0, 12.75, 18.0, 21.5}
    assert bins["price_bin_high"] == {25.5, 30.0, 33.25, 41.0, 120.0}


def test_distribution_bins_follow_the_shape_of_the_column(built):
    onto = built("dl_dist_bins")
    bins = _bins(onto, "price")

    assert bins == {
        "price_bin_very_low": {3.25},
        "price_bin_low": {7.5, 10.0},
        "price_bin_average": {12.75, 18.0, 21.5, 25.5, 30.0, 33.25},
        "price_bin_high": {41.0},
        "price_bin_very_high": {120.0},
    }


def test_values_and_bins_together_type_a_value_by_its_bin(built):
    onto = built("dl_bins")
    binned = {
        name
        for name, classes in onto.types.items()
        if classes & {"price_bin_low", "price_bin_high"}
    }

    assert binned
    literals = {v for p, s, v, _t in onto.data_values if s in binned}
    assert literals, "a binned value individual still carries its number"


def test_bins_alone_leave_the_number_off_the_value_individual(built):
    with_values = built("dl_bins")
    without = built("dl_bins_only")

    assert with_values.types == without.types, "the same values, in the same bins"
    assert {s for _, s, _v, _t in with_values.data_values if s.startswith("price_")}
    assert not {s for _, s, _v, _t in without.data_values if s.startswith("price_")}


def test_no_numeric_representation_leaves_the_numbers_out(built):
    onto = built("dl_no_numeric")

    assert "price" not in onto.declared
    assert "channel" in onto.classes, "only the numeric attributes are dropped"


def test_a_column_with_too_few_values_to_bin_keeps_its_numbers(built):
    """A class per distinct number would be a bin in name only."""
    # weight has four distinct values, below what a distribution fit needs.
    onto = built("dl_dist_bins")

    assert not any(name.startswith("weight_bin_") for name in onto.classes)
    assert "weight" in onto.classes
    assert {s for _, s, _v, _t in onto.data_values if s.startswith("weight_")}
    assert {"price_bin_low", "price_bin_average"} <= onto.classes


def test_an_attribute_is_cut_at_one_set_of_boundaries_everywhere(built):
    """Two tables carrying one attribute share its class, so a number cannot
    fall in one bin on an event and another on an object."""
    onto = built("dl_bins")
    bins = {c for c in onto.classes if c.startswith("price_bin_")}

    for individual, classes in onto.types.items():
        if individual.startswith("price_"):
            assert len(classes & bins) <= 1, individual


# --------------------------------------------------------------------------
# DECLARE membership
# --------------------------------------------------------------------------


def test_a_mined_constraint_is_labelled_with_what_it_says(tmp_path):
    """`Response__place_order__ship__` is safe in an IRI and unreadable, so a
    build that derives annotations writes the constraint out.
    """
    onto = emit(
        settings(
            order_representations=[
                EventOrderRepresentation.DF,
                EventOrderRepresentation.DECLARE,
            ],
            declare_constraints=declare_constraints(),
            derive_annotations=True,
        ),
        tmp_path / "declare.ofn",
    )
    labels = {name: text for name, kind, text in onto.annotations if kind == "label"}

    assert labels["Response__place_order__ship__"] == "Response (place order, ship)"
    assert labels["DeclareConstraint"] == "DECLARE constraint"


def test_a_trace_is_typed_by_the_constraints_it_satisfies(built):
    onto = built("dl_declare")
    satisfying = {
        t for t, c in onto.types.items() if "Response__place_order__ship__" in c
    }

    assert satisfying == {"o1"}, "o2 does not satisfy the constraint and is not typed"
    assert onto.parents["Response__place_order__ship__"] == {"DeclareConstraint"}


def test_each_case_concept_keeps_its_own_constraints(built):
    onto = built("dl_declare_multi_cc")

    assert {
        t for t, c in onto.types.items() if "Response__place_order__ship__" in c
    } == {"o1"}
    assert {t for t, c in onto.types.items() if "Existence__ship__" in c} == {"i1"}


def test_declare_without_constraints_is_refused():
    from semanticlog.onto.settings import EventOrderRepresentation

    config = settings(
        order_representations=[EventOrderRepresentation.DECLARE],
        declare_constraints=None,
    )

    with pytest.raises(ValueError, match="DECLARE"):
        config.inputs.validate(config.settings)


# --------------------------------------------------------------------------
# Inverses
# --------------------------------------------------------------------------


def test_implicit_inverses_assert_the_reverse_of_every_relation(built):
    onto = built("dl_inverses")
    asserted = {p for p, _, _ in onto.relations}
    reversed_properties = {p for p in asserted if p.startswith("inv_")}

    assert {"inv_concerns", "inv_contains", "inv_df_order"} <= reversed_properties
    for inverse in sorted(reversed_properties):
        forward = {
            (s, o) for p, s, o in onto.relations if p == inverse.removeprefix("inv_")
        }
        backward = {(s, o) for p, s, o in onto.relations if p == inverse}
        assert backward == {(o, s) for s, o in forward}, (
            f"{inverse} is not the reverse of {inverse.removeprefix('inv_')}"
        )


def test_an_inverse_that_reads_as_a_word_of_its_own_keeps_that_name(built):
    onto = built("dl_inverses")

    # `observed_at` reversed reads "observes", so it takes that name rather
    # than the inv_ prefix the rest get.
    observes = {(s, o) for p, s, o in onto.relations if p == "observes"}
    observed_at = {(s, o) for p, s, o in onto.relations if p == "observed_at"}

    assert observes == {(o, s) for s, o in observed_at}
    assert "inv_observed_at" not in onto.properties


def test_the_inverse_of_every_attribute_property_sits_under_attribute_of(built):
    onto = built("dl_inverses")
    reversed_attributes = {p for p in onto.properties if p.startswith("inv_has_")}

    assert reversed_attributes
    for prop in sorted(reversed_attributes):
        assert onto.property_parents[prop] == {"attribute_of"}


def test_axiomatic_inverses_relate_the_two_properties(built):
    assert ("concerns", "inv_concerns") in built("dl_inverses").inverses


def test_el_gets_the_assertions_but_not_the_axiom(built):
    onto = built("el_inverses")

    assert any(p == "inv_concerns" for p, _, _ in onto.relations)
    assert not onto.inverses


# The attribute history extension has its own file. This is the other half of
# it: nothing of the sort appears without the extension.
def test_no_history_is_emitted_without_the_extension(built):
    assert "AttributeChange" not in built("dl_df").declared


# --------------------------------------------------------------------------
# Settings that stop a build before it writes anything
# --------------------------------------------------------------------------


def test_bin_labels_have_to_match_the_bin_count():
    with pytest.raises(ValueError, match="bin_labels"):
        settings(bins=3, bin_labels=["low", "high"])


def test_the_two_binning_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        settings(
            data_representations=[
                DataRepresentation.EQUAL_BINS,
                DataRepresentation.DIST_BINS,
            ]
        ).settings.validate()
