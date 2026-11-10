"""`BuildInputs`: what a build reads from the log or from a file."""

from __future__ import annotations

import pytest

from semanticlog.core.enums import EventOrderRepresentation
from semanticlog.core.errors import ConfigError
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings


def test_case_concepts_given_as_one_string_are_rejected():
    with pytest.raises(ConfigError, match="case_concepts must be a list"):
        BuildInputs(case_concepts="order")


@pytest.mark.parametrize(
    ("order", "match"),
    [
        (EventOrderRepresentation.DF, "no case concepts were given"),
        (EventOrderRepresentation.DECLARE, "includes declare but no constraints"),
    ],
)
def test_an_order_representation_without_its_input_is_rejected(order, match):
    settings = OntoBuildSettings(order_representations=[order])

    with pytest.raises(ConfigError, match=match):
        BuildInputs().validate(settings)


def test_missing_inputs_are_reported_on_one_line_in_the_terms_a_user_types():
    settings = OntoBuildSettings(
        order_representations=[
            EventOrderRepresentation.DF,
            EventOrderRepresentation.DECLARE,
        ]
    )

    with pytest.raises(
        ConfigError, match="^The build inputs are incomplete: "
    ) as failure:
        BuildInputs().validate(settings)

    assert "\n" not in str(failure.value)
    assert "BuildInputs" not in str(failure.value)


def test_case_concepts_given_as_one_string_are_rejected_with_a_sentence():
    with pytest.raises(ConfigError, match="^The setting case_concepts must be a list"):
        BuildInputs(case_concepts="order")
