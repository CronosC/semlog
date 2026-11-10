"""The record of the individuals a build has asserted."""

from semlog.onto.emitted import EmittedIndividuals


def test_the_values_recorded_for_an_attribute_are_kept_apart_from_other_attributes():
    emitted = EmittedIndividuals()
    emitted.values_of("price").add("price_10")

    assert emitted.values_of("price") == {"price_10"}
    assert emitted.values_of("weight") == set()


def test_the_counts_sum_each_kind_over_every_attribute():
    emitted = EmittedIndividuals()
    emitted.values_of("price").update({"price_10", "price_20"})
    emitted.values_of("weight").add("weight_1")
    emitted.bins_of("price").add("price_10")
    emitted.timestamps.add("t_2024")

    assert emitted.counts() == {"values": 3, "bins": 1, "timestamps": 1}
