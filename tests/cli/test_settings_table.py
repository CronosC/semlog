"""The table of build settings: defaults, value readers, and expanding the input setting."""

from __future__ import annotations

import pytest

from semlog.cli.settings_table import (
    BUILD_SETTINGS,
    expand_inputs,
    read_setting,
    setting,
)
from semlog.core.errors import ConfigError
from semlog.onto.settings import OntoBuildSettings


def test_every_settings_field_has_a_row_with_the_same_default():
    written = OntoBuildSettings().to_dict()
    declared = {s.key: s.default for s in BUILD_SETTINGS if s.target == "settings"}

    assert declared == written


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


@pytest.mark.parametrize(
    ("key", "value", "minimum"),
    [("auto_case_concepts", 0, 1), ("max_ef_pairs", -3, 0), ("size_limit_mb", -1, 0)],
)
def test_a_whole_number_below_its_minimum_is_rejected(key, value, minimum):
    with pytest.raises(
        ConfigError, match=f"{key}: expected a whole number of at least {minimum}"
    ):
        setting(key).read(value, key)


@pytest.mark.parametrize(
    ("key", "minimum"),
    [("auto_case_concepts", 1), ("max_ef_pairs", 0), ("size_limit_mb", 0)],
)
def test_a_whole_number_at_its_minimum_is_accepted(key, minimum):
    assert setting(key).read(minimum, key) == minimum


@pytest.mark.parametrize(
    ("key", "value"),
    [("iri", "https://x.org/a b"), ("tbox_iri", "https://x.org/a|b#")],
)
def test_an_iri_setting_with_a_character_an_iri_cannot_hold_is_rejected(key, value):
    with pytest.raises(ConfigError, match=f"^{key}: .* cannot be written"):
        setting(key).read(value, key)


def test_an_iri_ending_in_a_hash_is_rejected_when_read():
    with pytest.raises(ConfigError, match="^--iri: must end in '/'"):
        setting("iri").read("https://x.org/log#", "--iri")


def test_a_value_given_as_a_flag_is_named_by_the_flag():
    with pytest.raises(ConfigError, match="^--profile: unknown value 'foo'"):
        read_setting({"profile": "foo"}, "profile", flagged={"profile"})


def test_a_value_from_the_config_is_named_by_its_key():
    with pytest.raises(ConfigError, match="^profile: unknown value 'foo'"):
        read_setting({"profile": "foo"}, "profile", flagged=set())


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


def test_inputs_sharing_a_stem_are_named_on_one_line(tmp_path, ocel):
    for directory in ("first", "second"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "log.jsonocel").write_bytes(ocel.read_bytes())

    with pytest.raises(ConfigError, match="share a stem") as failure:
        expand_inputs(
            [
                str(tmp_path / "first" / "log.jsonocel"),
                str(tmp_path / "second" / "log.jsonocel"),
            ]
        )

    assert "\n" not in str(failure.value)
