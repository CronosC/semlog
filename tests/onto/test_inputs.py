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
