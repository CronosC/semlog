"""Argument parsing, help text, and the flags that reach the settings.

Every build flag defaults to None, so a flag that was not given is told apart from one given its default.
"""

from __future__ import annotations

import argparse
import logging
import re

import pytest

from semanticlog import __version__
from semanticlog.cli import build_parser
from semanticlog.cli.parser import _log_level, flag_overrides
from semanticlog.cli.settings_table import BUILD_DEFAULTS, BUILD_SETTINGS
from semanticlog.core.errors import ConfigError

SUBCOMMANDS = ("build", "inspect", "mine-declare", "reason")


def parse(*argv: str):
    return build_parser().parse_args(list(argv))


def _help(command: str) -> str:
    sub = next(
        a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)
    )
    return sub.choices[command].format_help()


def test_every_subcommand_renders_its_help(capsys):
    for command in SUBCOMMANDS:
        with pytest.raises(SystemExit) as exit_code:
            parse(command, "--help")
        assert exit_code.value.code == 0
        assert capsys.readouterr().out.startswith(f"usage: semlog {command}")


def test_the_version_is_the_one_provenance_records(capsys):
    with pytest.raises(SystemExit):
        parse("--version")

    assert capsys.readouterr().out.strip() == f"semlog {__version__}"


def test_a_build_flag_that_was_not_given_stays_none():
    given = {
        key
        for key, value in vars(parse("build", "log.json")).items()
        if value is not None
    }

    assert given == {"input", "command", "dry_run"}


def test_only_the_flags_that_were_given_become_overrides():
    args = parse("build", "log.json", "--profile", "OWL2EL", "--bins", "4")

    assert flag_overrides(args) == {
        "input": ["log.json"],
        "profile": "OWL2EL",
        "bins": 4,
    }


def test_an_empty_positional_input_falls_through_to_the_config():
    assert "input" not in flag_overrides(parse("build", "--config", "run.toml"))


def test_a_list_flag_may_be_given_with_no_values():
    assert (
        parse("build", "log.json", "--order-representations").order_representations
        == []
    )


def test_a_boolean_flag_has_a_negated_form():
    assert parse("build", "log.json", "--materialize").materialize is True
    assert parse("build", "log.json", "--no-materialize").materialize is False


def test_reason_takes_its_backend_default_from_the_table():
    args = parse("reason", "o.owl")

    assert args.reasoner_backend == BUILD_DEFAULTS["reasoner_backend"]
    assert not hasattr(args, "profile")


def test_mine_declare_output_is_not_a_build_setting():
    args = parse("mine-declare", "log.json", "--output", "x.declare.json")

    assert args.declare_output == "x.declare.json"
    assert "output" not in flag_overrides(args)


@pytest.mark.parametrize(
    ("value", "problem"), [("50", "above 0 and up to 1"), ("half", "expected a number")]
)
def test_a_support_flag_that_is_not_a_share_is_rejected(value, problem, capsys):
    with pytest.raises(SystemExit):
        parse("mine-declare", "log.json", "--constraint-support", value)

    assert problem in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "level"),
    [
        (("-v", "build", "log.json"), logging.DEBUG),
        (("build", "log.json", "-v"), logging.DEBUG),
        (("build", "log.json"), logging.INFO),
    ],
)
def test_verbosity_is_read_on_either_side_of_the_command(argv, level):
    assert _log_level(parse(*argv)) == level


def test_quiet_and_verbose_together_are_rejected(capsys):
    with pytest.raises(SystemExit):
        parse("build", "log.json", "-q", "-v")

    assert "not allowed with" in capsys.readouterr().err


def test_quiet_and_verbose_on_either_side_of_the_command_are_rejected():
    with pytest.raises(ConfigError, match="cannot be combined"):
        _log_level(parse("-v", "build", "log.json", "-q"))


def test_every_setting_flag_states_its_default():
    assert [
        s.flag
        for s in BUILD_SETTINGS
        if s.argparse is not None and "(default:" not in s.help
    ] == []


def test_help_lists_each_enum_member_once():
    text = re.sub(r"\s+", " ", _help("build"))

    assert "df | ef | ef_explicit | declare" in text
    assert "DF | EF" not in text


def test_every_configurable_setting_has_a_flag_on_the_command_that_reads_it():
    flags = set(vars(parse("build", "log.json"))) | set(
        vars(parse("mine-declare", "log.json"))
    )

    assert set(BUILD_DEFAULTS) - flags == set()
