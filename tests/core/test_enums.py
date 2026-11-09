"""Reading a build setting written as text."""

import pytest

from semanticlog.core.enums import OutputFormat, OWLProfile, resolve_reasoner_backend
from semanticlog.core.errors import ConfigError


@pytest.mark.parametrize("written", [" owl2el ", OWLProfile.OWL2EL])
def test_a_setting_is_read_from_its_text_in_any_case_or_from_its_member(written):
    assert OWLProfile.parse(written, key="profile") is OWLProfile.OWL2EL


def test_an_unknown_setting_is_rejected_listing_the_choices():
    with pytest.raises(
        ConfigError, match=r"reasoner_backend: unknown value 'Fact\+\+'\. Choose from: "
    ):
        resolve_reasoner_backend("Fact++", key="reasoner_backend")


def test_an_output_format_is_read_from_the_suffix_of_a_path():
    assert OutputFormat.of_path("out/log.OWL") is OutputFormat.OWL


def test_a_path_whose_suffix_names_no_output_format_is_rejected():
    with pytest.raises(
        ConfigError, match=r"log\.txt.*\.ofn, \.owl, \.owx, \.ttl, \.omn"
    ):
        OutputFormat.of_path("out/log.txt")
