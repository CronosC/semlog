"""The table of build settings: defaults, value readers, and expanding the input setting."""

from __future__ import annotations

import pytest

from semanticlog.cli.settings_table import (
    BUILD_SETTINGS,
    enum_choices,
    expand_inputs,
    setting,
)
from semanticlog.core.enums import EventOrderRepresentation, OCELExtension
from semanticlog.core.errors import ConfigError
from semanticlog.onto.settings import OntoBuildSettings


def test_every_settings_field_has_a_row_with_the_same_default():
    written = OntoBuildSettings().to_dict()
    declared = {s.key: s.default for s in BUILD_SETTINGS if s.target == "settings"}

    assert declared == written


def test_enum_choices_list_each_member_once():
    assert enum_choices(EventOrderRepresentation) == "df | ef | ef_explicit | declare"
    assert enum_choices(OCELExtension) == "attribute_history"


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("bins", "five", "bins: expected a whole number, got 'five'"),
        ("materialize", "yes", "materialize: expected true or false, got 'yes'"),
        ("iri", 3, "iri: expected a string, got 3"),
        (
            "constraint_support",
            "0.5",
            "constraint_support: expected a number, got '0.5'",
        ),
        (
            "itemset_support",
            1.5,
            "itemset_support: expected a share of the traces above 0 and up to 1, got 1.5",
        ),
    ],
)
def test_a_value_of_the_wrong_sort_names_its_setting(key, value, match):
    with pytest.raises(ConfigError, match=match):
        setting(key).read(value, key)


def test_a_reasoner_backend_is_read_in_any_case():
    assert setting("reasoner_backend").read("pellet", "reasoner_backend") == "Pellet"


def test_a_directory_contributes_every_event_log_in_it_once(tmp_path, ocel):
    for name in ("a.jsonocel", "b.jsonocel"):
        (tmp_path / name).write_bytes(ocel.read_bytes())
    (tmp_path / "notes.md").write_text("ignored")

    inputs = expand_inputs([str(tmp_path), str(tmp_path / "a.jsonocel")])

    assert [p.name for p in inputs] == ["a.jsonocel", "b.jsonocel"]


def test_no_input_is_rejected():
    with pytest.raises(ConfigError, match="No input given"):
        expand_inputs([])


def test_a_missing_input_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="Input not found"):
        expand_inputs([str(tmp_path / "absent.json")])


def test_a_file_that_is_not_an_event_log_is_rejected(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("nothing to build")

    with pytest.raises(ConfigError, match="is not an event log semlog reads"):
        expand_inputs([str(path)])


def test_a_directory_without_event_logs_is_rejected(tmp_path):
    (tmp_path / "notes.md").write_text("nothing to build")

    with pytest.raises(ConfigError, match="contains no event logs"):
        expand_inputs([str(tmp_path)])


def test_two_inputs_sharing_a_stem_are_rejected(tmp_path, ocel):
    for directory in ("first", "second"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "log.jsonocel").write_bytes(ocel.read_bytes())

    with pytest.raises(ConfigError, match="These inputs share a stem"):
        expand_inputs(
            [
                str(tmp_path / "first" / "log.jsonocel"),
                str(tmp_path / "second" / "log.jsonocel"),
            ]
        )
