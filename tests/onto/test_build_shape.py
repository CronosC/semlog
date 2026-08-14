"""What each build configuration emits.

The table below is the record of the product. A change to what a build emits
fails here, naming the configuration that moved and the number or name that
moved with it. Counts say how much came out, and the two name sets say what
the configuration means: the entities it exists to produce, and the ones it
must leave out.

When a change to the emitter moves a number here, update the row and say in
the commit message which change moved it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tests.support.settings import NAMES

# The upper schema, which every configuration emits whatever else it does.
TBOX = frozenset(
    {
        "Event",
        "Object",
        "Attribute",
        "ObjectAttribute",
        "EventAttribute",
        "Time",
        "observed_at",
        "event_object_relation",
        "object_object_relation",
        "has_attribute",
        "has_data_value",
    }
)


@dataclass(frozen=True)
class Shape:
    """How much one configuration emits, and which names say what it is.

    Attributes:
        classes: Declared classes.
        object_properties: Declared object properties.
        data_properties: Declared data properties.
        individuals: Names asserted into a class.
        axioms: Axiom lines written.
        must_contain: Names this configuration exists to produce.
        must_not_contain: Names another configuration produces and this one
            must not.
        transitive: Properties this configuration asserts transitive.
        inverse_pairs: How many axiomatic inverse pairs it declares.
    """

    classes: int
    object_properties: int
    data_properties: int
    individuals: int
    axioms: int
    must_contain: frozenset[str] = field(default_factory=frozenset)
    must_not_contain: frozenset[str] = field(default_factory=frozenset)
    transitive: frozenset[str] = field(default_factory=frozenset)
    inverse_pairs: int = 0

    def counts(self) -> dict[str, int]:
        return {
            "classes": self.classes,
            "object_properties": self.object_properties,
            "data_properties": self.data_properties,
            "individuals": self.individuals,
            "axioms": self.axioms,
        }


# Names that mark one representation, used to say what each row leaves out.
_DF_PROPERTIES = frozenset({"df", "df_order"})
# Any trace projection bounds its traces, so ef_explicit on its own emits these
# without emitting directly-follows.
_TRACE_BOUNDARY = frozenset({"StartEvent_order", "EndEvent_order"})
_DF = _DF_PROPERTIES | _TRACE_BOUNDARY
_EF = frozenset({"ef", "ef_order"})
_INVERSES = frozenset({"inv_concerns", "inv_contains", "observes", "attribute_of"})
_BINS = frozenset({"price_bin_low", "price_bin_high"})
_HISTORY = frozenset({"AttributeChange", "changed_at", "attribute_change_value"})
_DECLARE = frozenset({"DeclareConstraint", "Response__place_order__ship__"})
_NUMERIC = frozenset({"price", "amount", "weight", "has_price"})


EXPECTED: dict[str, Shape] = {
    "dl_df": Shape(
        classes=18,
        object_properties=15,
        data_properties=1,
        individuals=90,
        axioms=373,
        must_contain=_DF,
        must_not_contain=_EF | _INVERSES | _BINS | _HISTORY | _DECLARE,
    ),
    "el_df": Shape(
        classes=18,
        object_properties=15,
        data_properties=1,
        individuals=90,
        axioms=369,
        must_contain=_DF,
        must_not_contain=_EF | _INVERSES | _BINS | _HISTORY | _DECLARE,
    ),
    "dl_df_ef": Shape(
        classes=18,
        object_properties=17,
        data_properties=1,
        individuals=90,
        axioms=400,
        must_contain=_DF | _EF,
        must_not_contain=_INVERSES | _BINS | _HISTORY | _DECLARE,
        # EF left to a reasoner is a transitivity axiom rather than assertions.
        transitive=frozenset({"ef_order"}),
    ),
    "dl_ef_explicit": Shape(
        classes=18,
        object_properties=17,
        data_properties=1,
        individuals=90,
        axioms=404,
        must_contain=_DF | _EF,
        must_not_contain=_INVERSES | _BINS | _HISTORY | _DECLARE,
    ),
    "el_ef_explicit": Shape(
        classes=18,
        object_properties=15,
        data_properties=1,
        individuals=90,
        axioms=387,
        must_contain=_EF | _TRACE_BOUNDARY,
        must_not_contain=_DF_PROPERTIES | _INVERSES | _BINS | _HISTORY | _DECLARE,
    ),
    "dl_inverses": Shape(
        classes=18,
        object_properties=30,
        data_properties=1,
        individuals=90,
        axioms=718,
        must_contain=_DF | _INVERSES,
        must_not_contain=_EF | _BINS | _HISTORY | _DECLARE,
        inverse_pairs=15,
    ),
    "el_inverses": Shape(
        classes=18,
        object_properties=30,
        data_properties=1,
        individuals=90,
        axioms=511,
        must_contain=_DF | _INVERSES,
        must_not_contain=_EF | _BINS | _HISTORY | _DECLARE,
        # EL has no InverseObjectProperties axiom, so the pairs are only asserted.
        inverse_pairs=0,
    ),
    "dl_bins": Shape(
        classes=24,
        object_properties=15,
        data_properties=1,
        individuals=90,
        axioms=412,
        must_contain=_DF | _BINS | frozenset({"price"}),
        must_not_contain=_EF | _INVERSES | _HISTORY | _DECLARE,
    ),
    "dl_bins_only": Shape(
        classes=24,
        object_properties=15,
        data_properties=1,
        individuals=90,
        axioms=385,
        must_contain=_DF | _BINS,
        must_not_contain=_EF | _INVERSES | _HISTORY | _DECLARE,
    ),
    "dl_dist_bins": Shape(
        classes=26,
        object_properties=15,
        data_properties=1,
        individuals=90,
        axioms=412,
        must_contain=_DF
        | frozenset(
            {"price_bin_low", "price_bin_average", "price_bin_very_high", "price"}
        ),
        # weight has four distinct values, too few to bin, so it keeps its
        # numbers rather than getting a class per number.
        must_not_contain=_EF
        | _INVERSES
        | _HISTORY
        | _DECLARE
        | frozenset({"weight_bin_0.75", "weight_bin_3.25"}),
    ),
    "dl_no_case_concept": Shape(
        classes=16,
        object_properties=13,
        data_properties=1,
        individuals=90,
        axioms=332,
        must_contain=frozenset({"price", "channel"}),
        must_not_contain=_DF | _EF | _INVERSES | _BINS | _HISTORY | _DECLARE,
    ),
    "dl_no_numeric": Shape(
        classes=15,
        object_properties=12,
        data_properties=1,
        individuals=63,
        axioms=274,
        must_contain=_DF | frozenset({"channel", "clerk"}),
        must_not_contain=_NUMERIC | _EF | _INVERSES | _BINS | _HISTORY | _DECLARE,
    ),
    "dl_history": Shape(
        classes=21,
        object_properties=20,
        data_properties=1,
        individuals=95,
        axioms=592,
        must_contain=_DF | _HISTORY,
        must_not_contain=_EF | _INVERSES | _BINS | _DECLARE,
    ),
    "dl_history_bins": Shape(
        classes=27,
        object_properties=20,
        data_properties=1,
        individuals=95,
        axioms=425,
        must_contain=_DF | _HISTORY | _BINS,
        must_not_contain=_EF | _INVERSES | _DECLARE,
    ),
    "dl_declare": Shape(
        classes=20,
        object_properties=15,
        data_properties=1,
        individuals=90,
        axioms=377,
        must_contain=_DF | _DECLARE,
        must_not_contain=_EF | _INVERSES | _BINS | _HISTORY,
    ),
    "dl_declare_multi_cc": Shape(
        classes=23,
        object_properties=16,
        data_properties=1,
        individuals=90,
        axioms=397,
        must_contain=_DF
        | _DECLARE
        | frozenset({"df_item", "StartEvent_item", "Existence__ship__"}),
        must_not_contain=_EF | _INVERSES | _BINS | _HISTORY,
    ),
}


def test_the_table_covers_every_configuration():
    assert sorted(EXPECTED) == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_the_build_emits_the_recorded_number_of_entities(name, built):
    assert built(name).counts() == EXPECTED[name].counts()


@pytest.mark.parametrize("name", NAMES)
def test_the_build_emits_the_names_the_configuration_is_for(name, built):
    missing = EXPECTED[name].must_contain - built(name).declared
    assert not missing, f"{name} should declare {sorted(missing)}"


@pytest.mark.parametrize("name", NAMES)
def test_the_build_leaves_out_the_names_the_configuration_excludes(name, built):
    present = EXPECTED[name].must_not_contain & built(name).declared
    assert not present, f"{name} should not declare {sorted(present)}"


@pytest.mark.parametrize("name", NAMES)
def test_the_build_emits_the_upper_schema(name, built):
    missing = TBOX - built(name).declared
    assert not missing, f"{name} is missing schema names {sorted(missing)}"


@pytest.mark.parametrize("name", NAMES)
def test_the_build_asserts_the_recorded_property_characteristics(name, built):
    onto = built(name)
    assert onto.transitive == set(EXPECTED[name].transitive)
    assert len(onto.inverses) == EXPECTED[name].inverse_pairs
