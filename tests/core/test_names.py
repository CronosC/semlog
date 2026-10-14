"""The name space every claimant is allocated out of."""

from itertools import islice

import pytest

from semanticlog.core.names import NameRegistry, candidates


def _has(name: str) -> set[str]:
    return {f"has_{name}"}


def test_a_claimant_is_offered_its_bare_name_then_its_prefixed_one_then_a_counter():
    offered = list(islice(candidates("ship", "event_"), 4))

    assert offered == ["ship", "event_ship", "event_ship_2", "event_ship_3"]


def test_the_first_claimant_keeps_a_name_and_the_second_takes_its_prefix():
    registry = NameRegistry()

    first = registry.allocate("order", "object type", prefix="object_")
    second = registry.allocate("order", "activity", prefix="event_")

    assert (first, second) == ("order", "event_order")


def test_a_reserved_name_whose_prefixed_form_is_taken_is_numbered_past():
    registry = NameRegistry({"Event", "object_Event"})

    assert registry.allocate("Event", "object type", prefix="object_") == (
        "object_Event_2"
    )


def test_a_claim_takes_the_names_derived_from_it_for_its_owner():
    registry = NameRegistry()

    registry.allocate("price", "attribute", prefix="attribute_", derives=_has)

    assert registry.owner("has_price") == "attribute"


def test_a_candidate_whose_derived_name_is_taken_is_passed_over():
    registry = NameRegistry({"has_price"})

    free = registry.allocate("price", "attribute", prefix="attribute_", derives=_has)

    assert free == "attribute_price"


@pytest.mark.parametrize(
    ("name", "pending", "expected"),
    [
        ("Event", {"Event", "object_Event"}, "object_Event_2"),
        ("order", {"order", "item"}, "order"),
    ],
    ids=["held_for_another", "held_for_itself"],
)
def test_a_name_pending_for_a_later_claimant_is_not_taken_from_it(
    name, pending, expected
):
    registry = NameRegistry({"Event"})

    assert registry.allocate(
        name, "object type", prefix="object_", pending=pending
    ) == (expected)


def test_a_candidate_the_claimant_cannot_be_emitted_under_is_passed_over():
    registry = NameRegistry()

    free = registry.allocate(
        "2fast", "relation", prefix="relation_", accepts=lambda n: n[0].isalpha()
    )

    assert free == "relation_2fast"


def test_a_prefix_no_candidate_passes_is_rejected():
    registry = NameRegistry()

    with pytest.raises(ValueError, match="Prefix '2_' gives '2_2fast'"):
        registry.allocate(
            "2fast", "relation", prefix="2_", accepts=lambda n: n[0].isalpha()
        )


def test_the_claimed_names_under_a_prefix_leave_out_the_others():
    registry = NameRegistry({"price_bin_low", "price_bin_high", "has_price"})

    assert registry.starting_with("price_") == {"price_bin_low", "price_bin_high"}
