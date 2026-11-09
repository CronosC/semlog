"""Reading a TOML config, following `extends`, and rendering a value back as TOML."""

from __future__ import annotations

import tomllib

import pytest

from semanticlog.cli.toml import load_config_file, toml_value
from semanticlog.core.enums import OWLProfile
from semanticlog.core.errors import ConfigError


def test_a_config_is_read_as_the_settings_it_names(write_config):
    path = write_config(
        "run.toml", 'profile = "OWL2EL"\norder_representations = ["df"]\n'
    )

    assert load_config_file(path) == {
        "profile": "OWL2EL",
        "order_representations": ["df"],
    }


def test_an_unknown_key_is_rejected_with_the_nearest_setting(write_config):
    path = write_config("run.toml", 'proflie = "OWL2EL"\n')

    with pytest.raises(ConfigError, match="'proflie' \\(did you mean 'profile'\\?\\)"):
        load_config_file(path)


def test_an_unknown_key_is_reported_on_one_line(write_config):
    path = write_config("run.toml", 'proflie = "OWL2EL"\n')

    with pytest.raises(ConfigError, match="unknown setting") as failure:
        load_config_file(path)

    assert "\n" not in str(failure.value)


def test_a_renamed_key_is_rejected_with_its_current_name(write_config):
    path = write_config("run.toml", "order = []\n")

    with pytest.raises(ConfigError, match="did you mean 'order_representations'"):
        load_config_file(path)


def test_data_representations_is_reported_as_renamed(write_config):
    path = write_config("run.toml", 'data_representations = ["values"]\n')

    with pytest.raises(ConfigError, match="did you mean 'attribute_representations'"):
        load_config_file(path)


def test_a_removed_key_is_reported_with_what_replaces_it(write_config):
    path = write_config("run.toml", "shared_attributes = true\n")

    with pytest.raises(
        ConfigError,
        match="'shared_attributes' was removed. Every attribute name is shared by "
        "the types that carry it. Delete the key.",
    ):
        load_config_file(path)


def test_extends_is_suggested_for_a_misspelling(write_config):
    path = write_config("run.toml", 'extend = "_base.toml"\n')

    with pytest.raises(ConfigError, match="did you mean 'extends'"):
        load_config_file(path)


def test_a_missing_config_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config_file(tmp_path / "absent.toml")


def test_invalid_toml_is_rejected(write_config):
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config_file(write_config("run.toml", "profile = \n"))


def test_extends_merges_the_base_and_lets_the_child_win(write_config):
    write_config("_base.toml", 'profile = "OWL2DL"\nbins = 7\n')
    child = write_config("run.toml", 'extends = "_base.toml"\nprofile = "OWL2EL"\n')

    assert load_config_file(child) == {"profile": "OWL2EL", "bins": 7}


def test_extends_resolves_relative_to_the_file_it_appears_in(tmp_path, write_config):
    write_config("_base.toml", 'profile = "OWL2EL"\n')
    (tmp_path / "nested").mkdir()
    child = write_config("nested/run.toml", 'extends = "../_base.toml"\n')

    assert load_config_file(child) == {"profile": "OWL2EL"}


def test_an_extends_cycle_is_rejected(write_config):
    write_config("a.toml", 'extends = "b.toml"\n')
    write_config("b.toml", 'extends = "a.toml"\n')

    with pytest.raises(ConfigError, match="'extends' forms a cycle"):
        load_config_file(write_config("c.toml", 'extends = "a.toml"\n'))


def test_extends_that_is_not_a_string_is_rejected(write_config):
    with pytest.raises(ConfigError, match="'extends' must be a path string"):
        load_config_file(write_config("run.toml", "extends = 3\n"))


def test_a_rendered_value_reads_back_as_the_same_value():
    values = {
        "enum": OWLProfile.OWL2EL,
        "flag": False,
        "count": 3,
        "share": 0.25,
        "items": ["a", 2],
        "path": 'say "hi"\\there\n',
    }
    text = "\n".join(f"{key} = {toml_value(value)}" for key, value in values.items())

    assert tomllib.loads(text) == {**values, "enum": "OWL2EL"}
