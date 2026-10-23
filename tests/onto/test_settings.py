from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from semanticlog.core.enums import (
    AttributeRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OWLProfile,
)
from semanticlog.core.errors import ConfigError
from semanticlog.core.vocabulary import TBOX_IRI
from semanticlog.onto.settings import OntoBuildSettings

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


def test_the_default_tbox_iri_is_the_constant():
    assert OntoBuildSettings().tbox_iri == TBOX_IRI


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


def test_an_unknown_enum_value_is_rejected_on_construction():
    with pytest.raises(ConfigError, match="profile: unknown value 'OWL2XL'"):
        OntoBuildSettings(profile="OWL2XL")


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
