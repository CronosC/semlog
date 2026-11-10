"""`OntoBuildSettings`: the settings of a build."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from semlog.core.enums import (
    AttributeRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OWLProfile,
)
from semlog.core.errors import ConfigError
from semlog.onto.settings import OntoBuildSettings

DF = EventOrderRepresentation.DF
EF = EventOrderRepresentation.EF


def test_settings_cannot_be_changed_in_place():
    settings = OntoBuildSettings(bin_labels=["low", "high"], bins=2)

    with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
        settings.derive_annotations = True
    with pytest.raises(AttributeError, match="has no attribute 'append'"):
        settings.bin_labels.append("higher")


def test_a_list_passed_in_is_not_shared_with_the_settings():
    labels = ["low", "high"]
    settings = OntoBuildSettings(bin_labels=labels, bins=2)

    labels.append("higher")

    assert settings.bin_labels == ("low", "high")


def test_replace_returns_a_copy_and_leaves_the_original_alone():
    settings = OntoBuildSettings(bin_labels=["low", "high"], bins=2)

    changed = settings.replace(profile=OWLProfile.OWL2EL, bin_labels=["a", "b"])

    assert (settings.profile, settings.bin_labels) == (
        OWLProfile.OWL2DL,
        ("low", "high"),
    )
    assert (changed.profile, changed.bin_labels) == (OWLProfile.OWL2EL, ("a", "b"))


def test_an_enum_setting_may_be_written_as_its_value_or_its_member_name():
    settings = OntoBuildSettings(
        profile="OWL2EL", attribute_representations=["equal_bins", "INDIVIDUALS"]
    )

    assert settings.profile is OWLProfile.OWL2EL
    assert settings.attribute_representations == (
        AttributeRepresentation.EQUAL_BINS,
        AttributeRepresentation.INDIVIDUALS,
    )
    assert settings.to_dict()["attribute_representations"] == [
        "equal_bins",
        "individuals",
    ]


@pytest.mark.parametrize(
    ("iri", "match"),
    [
        ("https://example.org/schema", "must end in '#' or '/'"),
        ("", "cannot be written"),
        ("https://exa mple.org/#", "cannot be written"),
    ],
    ids=["no terminator", "empty", "whitespace"],
)
def test_a_tbox_iri_that_cannot_hold_the_schema_is_rejected(iri, match):
    with pytest.raises(ConfigError, match=match):
        OntoBuildSettings(tbox_iri=iri)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"profile": OWLProfile.OWL2EL, "order_representations": [DF, EF]},
            "ef is not available under OWL2EL",
        ),
        ({"order_representations": [EF]}, "ef needs df"),
        (
            {
                "profile": OWLProfile.OWL2EL,
                "inverse_representations": [InverseRepresentation.AXIOMATIC],
            },
            "axiomatic inverses .* are not available under OWL2EL",
        ),
        (
            {
                "attribute_representations": [
                    AttributeRepresentation.INDIVIDUALS,
                    AttributeRepresentation.EQUAL_BINS,
                    AttributeRepresentation.DIST_BINS,
                ],
                "bins": 2,
                "bin_labels": ["low", "high"],
            },
            "equal_bins and dist_bins cannot both be set",
        ),
        ({"attribute_representations": []}, "attribute_representations is empty"),
        (
            {
                "attribute_representations": [AttributeRepresentation.EQUAL_BINS],
                "bins": 2,
                "bin_labels": ["low", "high"],
            },
            "equal_bins needs individuals",
        ),
    ],
    ids=[
        "ef under el",
        "ef without df",
        "axiomatic inverses under el",
        "two binnings",
        "no representation",
        "bins without individuals",
    ],
)
def test_settings_that_contradict_each_other_fail_validation(overrides, match):
    with pytest.raises(ConfigError, match=match):
        OntoBuildSettings(**overrides).validate()


def test_every_contradiction_is_reported_on_one_line():
    settings = OntoBuildSettings(profile=OWLProfile.OWL2EL, order_representations=[EF])

    with pytest.raises(
        ConfigError, match="^The settings contradict each other: "
    ) as failure:
        settings.validate()

    assert "\n" not in str(failure.value)
    assert "ef needs df" in str(failure.value)


@pytest.mark.parametrize("char", ['"', "{", "|", "\\", "^", "`"])
def test_a_tbox_iri_with_a_character_an_iri_cannot_hold_is_rejected(char):
    with pytest.raises(ConfigError, match="^tbox_iri: .* cannot be written"):
        OntoBuildSettings(tbox_iri=f"https://example.org/a{char}b#")


def test_a_sequence_given_as_one_string_is_rejected_with_a_sentence():
    with pytest.raises(
        ConfigError, match="^The setting order_representations must be a list"
    ):
        OntoBuildSettings(order_representations="df")
