"""Argument parsing, help text, and the flags reaching the settings.

Every build flag defaults to None so that a flag which was not given can be
told apart from one given its default value, which is what lets the config
file sit between the two. The completeness tests are the ones that notice a
setting gaining a name in one place and not the other.
"""

from __future__ import annotations

import dataclasses

import pytest

from semanticlog.cli import build_parser
from semanticlog.cli.config import BUILD_DEFAULTS
from semanticlog.cli.parser import build_overrides
from semanticlog.onto.settings import OntoBuildSettings

SUBCOMMANDS = ("build", "inspect", "mine-declare", "reason")


def parse(*argv: str):
    return build_parser().parse_args(list(argv))


def test_every_subcommand_is_reachable():
    help_text = build_parser().format_help()

    for command in SUBCOMMANDS:
        assert command in help_text


@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_each_subcommand_renders_its_help(command, capsys):
    """Help interpolates the defaults, so a bad one fails here rather than at use."""
    with pytest.raises(SystemExit) as exit_code:
        parse(command, "--help")

    assert exit_code.value.code == 0
    assert capsys.readouterr().out


def test_the_version_is_reported(capsys):
    with pytest.raises(SystemExit):
        parse("--version")

    assert "semlog" in capsys.readouterr().out


def test_the_reported_version_is_the_one_provenance_records(capsys):
    """One version source, so an .owl file cannot disagree with --version."""
    from semanticlog import __version__

    with pytest.raises(SystemExit):
        parse("--version")

    assert capsys.readouterr().out.strip() == f"semlog {__version__}"


# --------------------------------------------------------------------------
# Nothing is set unless it was asked for
# --------------------------------------------------------------------------


def test_a_build_flag_that_was_not_given_stays_none():
    args = parse("build", "log.json")

    given = {key for key, value in vars(args).items() if value is not None}
    assert given == {"input", "command", "verbose", "quiet", "dry_run"}


def test_only_the_flags_that_were_given_become_overrides():
    args = parse("build", "log.json", "--profile", "OWL2EL", "--bins", "4")

    assert build_overrides(args) == {
        "input": ["log.json"],
        "profile": "OWL2EL",
        "bins": 4,
    }


def test_an_empty_positional_input_falls_through_to_the_config():
    args = parse("build", "--config", "run.toml")

    assert "input" not in build_overrides(args)


def test_a_list_flag_may_be_given_with_no_values():
    assert (
        parse("build", "log.json", "--order-representations").order_representations
        == []
    )
    assert parse("build", "log.json", "--case-concepts").case_concepts == []


# --------------------------------------------------------------------------
# What the parser accepts and refuses
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["OWL2DL", "owl2dl", "Owl2Dl"])
def test_a_profile_is_accepted_in_either_spelling_and_any_case(spelling):
    assert parse("build", "log.json", "--profile", spelling).profile == spelling


@pytest.mark.parametrize("backend", ["HermiT", "hermit", "PELLET", "elk"])
def test_a_reasoner_backend_is_accepted_in_any_case(backend):
    assert parse("reason", "o.owl", "--backend", backend).backend in (
        "HermiT",
        "Pellet",
        "ELK",
    )


def test_an_unknown_reasoner_backend_is_refused(capsys):
    with pytest.raises(SystemExit):
        parse("reason", "o.owl", "--backend", "Konclude")

    assert "unknown value 'Konclude'" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "setting", "expected"),
    [
        ("--derive-annotations", "derive_annotations", True),
        ("--no-derive-annotations", "derive_annotations", False),
        ("--materialize", "materialize", True),
        ("--no-materialize", "materialize", False),
    ],
)
def test_a_boolean_flag_can_be_turned_off_as_well_as_on(flag, setting, expected):
    assert getattr(parse("build", "log.json", flag), setting) is expected


# --------------------------------------------------------------------------
# Every setting is reachable from both directions
# --------------------------------------------------------------------------


def test_every_configurable_setting_has_a_command_line_flag():
    flags = set(vars(parse("build", "log.json")))

    missing = set(BUILD_DEFAULTS) - flags
    assert not missing, (
        f"{sorted(missing)} can be set in a config but not on the command line"
    )


def test_every_build_setting_can_be_named_by_a_run():
    """Every one of them, with no exceptions.

    The settings hold only the choices a build is configured with, and each of
    those is a config key. What is read off a log or out of a file is a
    `BuildInputs`, which a run assembles rather than names.
    """
    fields = {f.name for f in dataclasses.fields(OntoBuildSettings)}

    unreachable = fields - set(BUILD_DEFAULTS)
    assert not unreachable, (
        f"{sorted(unreachable)} change what a build emits but no config key or "
        "flag sets them"
    )
