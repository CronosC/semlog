"""Argument parsing, help text, and the flags that reach the settings.

Every build flag defaults to None, so a flag that was not given is told apart from one given its default.
"""

from __future__ import annotations

import argparse
import logging
import re

import pytest

from semlog import __version__
from semlog.cli import EXIT_USAGE, build_parser, main
from semlog.cli.parser import _log_level, flag_overrides
from semlog.cli.settings_table import BUILD_DEFAULTS, BUILD_SETTINGS
from semlog.core.errors import ConfigError

SUBCOMMANDS = ("build", "inspect", "mine-declare", "reason")


def parse(*argv: str):
    return build_parser().parse_args(list(argv))


def _subparser(command: str) -> argparse.ArgumentParser:
    sub = next(
        a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)
    )
    return sub.choices[command]


def _help(command: str) -> str:
    return _subparser(command).format_help()


def test_every_subcommand_renders_its_help(capsys):
    for command in SUBCOMMANDS:
        with pytest.raises(SystemExit) as exit_code:
            parse(command, "--help")
        assert exit_code.value.code == 0
        assert capsys.readouterr().out.startswith(f"usage: semlog {command}")


def test_the_version_is_the_one_the_build_record_holds(capsys):
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


def test_mine_declare_writes_to_the_declare_file_flag():
    args = parse("mine-declare", "log.json", "--declare-file", "x.declare.json")

    assert flag_overrides(args)["declare_file"] == "x.declare.json"
    assert "output" not in flag_overrides(args)


def test_reason_takes_the_backend_flag_build_takes():
    assert parse("reason", "o.owl", "--reasoner-backend", "ELK").reasoner_backend == (
        "ELK"
    )


@pytest.mark.parametrize(
    "argv",
    [
        ("mine-declare", "log.json", "--output", "x.declare.json"),
        ("reason", "o.owl", "--backend", "ELK"),
    ],
)
def test_a_superseded_flag_is_rejected(argv, capsys):
    with pytest.raises(SystemExit) as exit_code:
        parse(*argv)

    assert exit_code.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_a_support_flag_that_is_not_a_number_is_rejected(capsys):
    with pytest.raises(SystemExit):
        parse("mine-declare", "log.json", "--constraint-support", "half")

    assert "invalid float value: 'half'" in capsys.readouterr().err


def test_a_support_flag_that_is_not_a_share_exits_two(ocel, capsys):
    argv = ["mine-declare", str(ocel), "--constraint-support", "50", "-q"]

    assert main(argv) == EXIT_USAGE
    assert (
        "--constraint-support: expected a share of the traces above 0 and up to 1"
        in (capsys.readouterr().err)
    )


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


def test_the_inspect_description_names_only_flags_inspect_takes():
    inspect = _subparser("inspect")
    taken = {flag for action in inspect._actions for flag in action.option_strings}

    assert set(re.findall(r"--[a-z-]+", inspect.description)) <= taken


def test_the_build_description_wraps_to_the_terminal_width(monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")

    assert max(len(line) for line in _help("build").splitlines()) <= 80


def test_the_epilog_names_every_file_a_build_can_write():
    epilog = build_parser().epilog

    for written in (".ofn", ".owl", "output_formats", ".backmap.json", "--log-dir"):
        assert written in epilog
