"""What each configuration of the matrix emits.

A change to what a build emits fails here, naming the configuration and the number or name that moved. When a change to the emitter moves a number, update the row and say in the commit message which change moved it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tests.support.settings import NAMES

# The upper schema, which every configuration emits.
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
        must_contain: Names this configuration exists to produce.
        must_not_contain: Names another configuration produces and this one must not.
        transitive: Properties asserted transitive.
        inverse_pairs: Axiomatic inverse pairs declared.
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


_DF_PROPERTIES = frozenset({"df", "df_order"})
# Any trace projection bounds its traces, so ef_explicit without df emits these.
_TRACE_BOUNDARY = frozenset({"StartEvent_order", "EndEvent_order"})
_DF = _DF_PROPERTIES | _TRACE_BOUNDARY
_EF = frozenset({"ef", "ef_order"})
_INVERSES = frozenset({"inv_concerns", "inv_contains", "observes", "attribute_of"})
_EQUAL_BINS = frozenset({"price_bin_low", "price_bin_high"})
_DIST_BINS = frozenset(
    {"price_bin_very_low", "price_bin_average", "price_bin_very_high"}
)
# weight takes four distinct values, too few for either mode to bin.
_NARROW_BINS = frozenset({"weight_bin_low", "weight_bin_high", "weight_bin_average"})
_HISTORY = frozenset(
    {"AttributeChange", "changed_at", "attribute_change_value", "caused_by"}
)
_DECLARE = frozenset({"DeclareConstraint", "Response__place_order__ship__"})
_NUMERIC = frozenset({"price", "amount", "weight", "has_price"})


EXPECTED: dict[str, Shape] = {
    "dl_df": Shape(
        classes=18,
        object_properties=15,
        data_properties=8,
        individuals=90,
        axioms=425,
        must_contain=_DF | _NUMERIC,
        must_not_contain=_EF | _INVERSES | _EQUAL_BINS | _HISTORY | _DECLARE,
    ),
    "dl_inverses": Shape(
        classes=18,
        object_properties=30,
        data_properties=8,
        individuals=90,
        axioms=784,
        must_contain=_DF | _INVERSES,
        must_not_contain=_EF | _EQUAL_BINS | _HISTORY | _DECLARE,
        inverse_pairs=15,
    ),
    "el_ef_explicit_bins": Shape(
        classes=22,
        object_properties=30,
        data_properties=8,
        individuals=90,
        axioms=800,
        must_contain=_EF | _TRACE_BOUNDARY | _INVERSES | _EQUAL_BINS | _NUMERIC,
        must_not_contain=_DF_PROPERTIES | _HISTORY | _DECLARE | _NARROW_BINS,
        # EL has no InverseObjectProperties axiom, so the pairs are only asserted.
        inverse_pairs=0,
    ),
    "dl_ef_dist_bins_history": Shape(
        classes=29,
        object_properties=23,
        data_properties=8,
        individuals=95,
        axioms=539,
        must_contain=_DF | _EF | _DIST_BINS | _HISTORY | _NUMERIC,
        must_not_contain=_INVERSES | _DECLARE | _NARROW_BINS,
        # EF left to a reasoner is a transitivity axiom.
        transitive=frozenset({"ef_order"}),
    ),
    "el_history_literals": Shape(
        classes=16,
        object_properties=13,
        data_properties=8,
        individuals=56,
        axioms=467,
        must_contain=_DF | _HISTORY | frozenset({"has_value_price"}),
        must_not_contain=_EF
        | _INVERSES
        | _DECLARE
        | _EQUAL_BINS
        | _DIST_BINS
        | frozenset({"price", "has_price", "change_value_price"}),
    ),
    "dl_declare_multi_cc": Shape(
        classes=23,
        object_properties=32,
        data_properties=8,
        individuals=90,
        axioms=900,
        must_contain=_DF
        | _DECLARE
        | _NUMERIC
        | frozenset(
            {"df_item", "StartEvent_item", "Existence__ship__", "inv_concerns"}
        ),
        must_not_contain=_EF | _EQUAL_BINS | _HISTORY,
    ),
    "dl_no_case_concept": Shape(
        classes=19,
        object_properties=38,
        data_properties=8,
        individuals=95,
        axioms=483,
        must_contain=_HISTORY | _INVERSES | _NUMERIC,
        must_not_contain=_DF | _EF | _EQUAL_BINS | _DECLARE,
        inverse_pairs=19,
    ),
}


def test_the_table_covers_every_configuration():
    assert sorted(EXPECTED) == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_the_build_emits_the_recorded_shape(name, built):
    onto = built(name)
    expected = EXPECTED[name]

    assert {
        **onto.counts(),
        "transitive": onto.transitive,
        "inverse_pairs": len(onto.inverses),
        "missing": (TBOX | expected.must_contain) - onto.declared,
        "excluded but present": expected.must_not_contain & onto.declared,
    } == {
        **expected.counts(),
        "transitive": set(expected.transitive),
        "inverse_pairs": expected.inverse_pairs,
        "missing": set(),
        "excluded but present": set(),
    }
