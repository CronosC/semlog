"""The argument parser and the flags each subcommand takes. Start at `build_parser`.

Every setting flag defaults to None. `flag_overrides` keeps the flags that were
passed, and only those override the config file. The defaults are in
`BUILD_DEFAULTS`.
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from collections.abc import Iterable

from semlog import __version__
from semlog.cli.settings_table import (
    BUILD_DEFAULTS,
    BUILD_SETTINGS,
    DECLARE_MINING_KEYS,
    LOG_INPUT_KEYS,
    setting,
)
from semlog.core.errors import ConfigError

# `input` is positional, so it has no flag.
_LOG_INPUT_FLAGS: tuple[str, ...] = tuple(k for k in LOG_INPUT_KEYS if k != "input")

_BUILD_ONLY_KEYS: tuple[str, ...] = tuple(
    s.key
    for s in BUILD_SETTINGS
    if s.key not in LOG_INPUT_KEYS and s.key not in DECLARE_MINING_KEYS
)

_INSPECT_EXCLUDED: tuple[str, ...] = ("case_concepts", "auto_case_concepts")


_EPILOG = textwrap.dedent("""\
    examples:
      semlog inspect log.jsonocel
      semlog build log.jsonocel --case-concepts Candidate Recruiter
      semlog build --config datasets/runs/hiring.toml
      semlog build --config datasets/runs/hiring.toml --dry-run > resolved.toml
      semlog mine-declare log.jsonocel --constraints Response Precedence
      semlog reason hiring.ofn --check

    a build writes one ontology per input into --output: <name>.ofn in
    functional syntax, or <name>.owl in RDF/XML under --no-streaming. Each
    format in output_formats adds <name>.<format>, save_backmap adds
    <name>.backmap.json, and --log-dir receives one log file per input. The
    semlog version, the resolved settings and the case concepts are annotated
    onto the ontology.
    """)


def _add_setting_flag(
    parser: argparse.ArgumentParser, key: str, *, help: str | None = None
) -> None:
    """Add the flag of one setting, with its own help unless `help` is given."""
    declared = setting(key)
    parser.add_argument(
        declared.flag,
        dest=key,
        default=None,
        help=help or declared.help,
        **(declared.argparse or {}),
    )


def _add_settings_flags(parser: argparse.ArgumentParser, keys: Iterable[str]) -> None:
    """Add a flag for each of the given settings."""
    for key in keys:
        _add_setting_flag(parser, key)


def _add_log_input_flags(
    parser: argparse.ArgumentParser,
    *,
    input_help: str | None = None,
    exclude: tuple[str, ...] = (),
) -> None:
    """Add the inputs, the config file, and the flags for reading a log."""
    parser.add_argument(
        "input",
        nargs="*",
        default=None,
        metavar="PATH",
        help=input_help or setting("input").help,
    )
    parser.add_argument(
        "--config",
        metavar="TOML",
        help=(
            "TOML config to read settings from. Flags override it. An 'extends' "
            "path in the config resolves relative to the config file."
        ),
    )
    _add_settings_flags(parser, (k for k in _LOG_INPUT_FLAGS if k not in exclude))


def _verbosity_parser() -> argparse.ArgumentParser:
    """Return the parent parser with `-v` and `-q`, shared by every command."""
    # A subparser default overwrites a flag given before the command name, so
    # the defaults are suppressed and `_log_level` supplies them.
    parent = argparse.ArgumentParser(add_help=False)
    verbosity = parent.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Log at DEBUG level.",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Log warnings and errors only.",
    )
    return parent


def _add_build_parser(
    sub: argparse._SubParsersAction, parent: argparse.ArgumentParser
) -> None:
    """Add the `build` subcommand and its flags."""
    build = sub.add_parser(
        "build",
        parents=[parent],
        help="Build one OWL ontology per event log.",
        description=(
            "Build one OWL ontology per event log. Settings come from a TOML "
            "config, from flags, or from both. A flag overrides the config. "
            "Relative paths in a config resolve against the current directory."
        ),
    )
    build.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration and the files it would write, then stop.",
    )
    _add_log_input_flags(build)
    _add_settings_flags(build, _BUILD_ONLY_KEYS)


def _add_inspect_parser(
    sub: argparse._SubParsersAction, parent: argparse.ArgumentParser
) -> None:
    """Add the `inspect` subcommand and its flags."""
    inspect = sub.add_parser(
        "inspect",
        parents=[parent],
        help="Summarize an event log without building anything.",
        description=(
            "Print the activities, object types, relations and time range of an "
            "event log. Names are shown as the build emits them. An activity "
            "or object type name that cleaning changed is listed with its "
            "source spelling."
        ),
    )
    _add_log_input_flags(inspect, exclude=_INSPECT_EXCLUDED)


def _add_mine_declare_parser(
    sub: argparse._SubParsersAction, parent: argparse.ArgumentParser
) -> None:
    """Add the `mine-declare` subcommand and its flags."""
    mine = sub.add_parser(
        "mine-declare",
        parents=[parent],
        help="Mine DECLARE constraints from an event log into a JSON file.",
        description=(
            "Mine DECLARE constraints from one event log and write them to a "
            "JSON file for 'semlog build --declare-file'. Requires the [declare] "
            "extra and the Lydia solver."
        ),
    )
    _add_setting_flag(
        mine,
        "declare_file",
        help=(
            "Where to write the constraints (default: the declare_file of "
            "--config, or <input stem>.declare.json)."
        ),
    )
    _add_settings_flags(mine, DECLARE_MINING_KEYS)
    _add_log_input_flags(
        mine,
        input_help="Event log to mine. Omit to use the 'input' key of --config.",
    )


def _add_reason_parser(
    sub: argparse._SubParsersAction, parent: argparse.ArgumentParser
) -> None:
    """Add the `reason` subcommand and its flags."""
    reason_parser = sub.add_parser(
        "reason",
        parents=[parent],
        help="Materialize or consistency-check an existing ontology.",
        description=(
            "Run a reasoner over a built ontology. Without --output the input "
            "file is overwritten with the materialized ontology."
        ),
    )
    reason_parser.add_argument(
        "onto", metavar="PATH", help="Ontology file to reason over."
    )
    _add_setting_flag(reason_parser, "reasoner_backend")
    # `reason` reads no config, so its flag holds the default itself.
    reason_parser.set_defaults(reasoner_backend=BUILD_DEFAULTS["reasoner_backend"])
    reason_parser.add_argument(
        "--output",
        metavar="PATH",
        help=(
            "Write the materialized ontology here, in the format its suffix "
            "names (default: overwrite the input)."
        ),
    )
    reason_parser.add_argument(
        "--check",
        action="store_true",
        help="Report whether the ontology is consistent and write nothing.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for the whole command line."""
    parent = _verbosity_parser()
    parser = argparse.ArgumentParser(
        prog="semlog",
        parents=[parent],
        description="Convert OCEL 2.0, CSV and XES event logs to OWL 2 ontologies.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"semlog {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    _add_build_parser(sub, parent)
    _add_inspect_parser(sub, parent)
    _add_mine_declare_parser(sub, parent)
    _add_reason_parser(sub, parent)
    return parser


def _log_level(args: argparse.Namespace) -> int:
    """Turn the verbosity flags into a logging level.

    Raises:
        ConfigError: If both `-v` and `-q` were given.
    """
    quiet = getattr(args, "quiet", False)
    verbose = getattr(args, "verbose", False)
    if quiet and verbose:
        raise ConfigError("-v/--verbose and -q/--quiet cannot be combined.")
    if quiet:
        return logging.WARNING
    return logging.DEBUG if verbose else logging.INFO


def flag_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Return the settings given on the command line.

    Arguments left at None are dropped, and so is an empty input list.
    """
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key in BUILD_DEFAULTS and value is not None
    }
    if overrides.get("input") == []:
        del overrides["input"]
    return overrides
