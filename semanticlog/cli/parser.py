"""The argument parser, and the flags each subcommand takes.

Every flag defaults to None. A flag that was not given has to be
distinguishable from one that was given its default value, because the config
file sits between the two and a flag only overrides it when it was passed. The
real defaults live in `BUILD_DEFAULTS`.
"""

from __future__ import annotations

import argparse
import logging
import textwrap

from semanticlog.cli.config import (
    BUILD_DEFAULTS,
    BUILD_SETTINGS,
)
from semanticlog.core.enums import (
    REASONER_BACKENDS,
    OCELExtension,
    OWLProfile,
    resolve_reasoner_backend,
)


def semlog_version() -> str:
    """Return the version the package reports, which provenance also records."""
    from semanticlog import __version__

    return __version__


def _reasoner_backend(value: str) -> str:
    """Accept a backend name in any case, and reject an unknown one."""
    try:
        return resolve_reasoner_backend(value, key="--backend")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


_EPILOG = textwrap.dedent("""\
    examples:
      semlog inspect datasets/ocel/Hiring.xml
      semlog build datasets/ocel/Hiring.xml --case-concepts Candidate Recruiter
      semlog build --config runs/hiring.toml
      semlog build --config runs/hiring.toml --dry-run
      semlog mine-declare datasets/ocel/enron.json --constraints Response Precedence
      semlog reason datasets/onto/hiring.owl --profile OWL2EL

    a build writes one file, <name>.owl, with the input checksum, the resolved
    settings, the command and the semlog version annotated onto the ontology.
    """)


def _verbosity_parser() -> argparse.ArgumentParser:
    """Return the parent parser holding the flags every subcommand shares."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log at DEBUG level.",
    )
    parent.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Log warnings and errors only.",
    )
    return parent


def _add_build_parser(sub, parent: argparse.ArgumentParser) -> None:
    """Add the `build` subcommand and its flags."""
    build = sub.add_parser(
        "build",
        parents=[parent],
        help="Build an OWL ontology from one or more OCEL files.",
        description=(
            "Build an OWL ontology from OCEL 2.0 event data. Settings may come "
            "from a TOML config, from flags, or from both; a flag always wins "
            "over the config. Relative paths in a config are resolved against "
            "the current directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    build.add_argument(
        "input",
        nargs="*",
        default=None,
        metavar="PATH",
        help=(
            "OCEL files or directories to build. A directory contributes every "
            "OCEL file in it. Omit to use the 'input' key of --config."
        ),
    )
    build.add_argument(
        "--config",
        metavar="TOML",
        help="TOML config to read settings from. Flags override it.",
    )
    build.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration and the files it would write, then stop.",
    )
    for setting in BUILD_SETTINGS:
        if setting.argparse is None:
            continue
        build.add_argument(
            setting.flag, default=None, help=setting.help, **setting.argparse
        )


def _add_inspect_parser(sub, parent: argparse.ArgumentParser) -> None:
    """Add the `inspect` subcommand and its flags."""
    inspect = sub.add_parser(
        "inspect",
        parents=[parent],
        help="Summarize an OCEL file without building anything.",
        description=(
            "Print the activities, object types, relations and time range of an "
            "OCEL file, plus the case concepts a build would auto-select. Names "
            "are shown in the cleaned form that --case-concepts expects."
        ),
    )
    inspect.add_argument(
        "input", nargs="+", metavar="PATH", help="OCEL files to summarize."
    )
    inspect.add_argument(
        "--suggest",
        type=int,
        default=3,
        metavar="N",
        help="How many case concepts to suggest (default: 3).",
    )
    inspect.add_argument(
        "--ocel-extensions",
        nargs="*",
        default=[],
        metavar="EXT",
        help=f"OCEL 2.0 extensions to load: {' | '.join(OCELExtension.spellings())}.",
    )


def _add_mine_declare_parser(sub, parent: argparse.ArgumentParser) -> None:
    """Add the `mine-declare` subcommand and its flags."""
    mine = sub.add_parser(
        "mine-declare",
        parents=[parent],
        help="Mine DECLARE constraints from an OCEL file into a CSV.",
        description=(
            "Mine DECLARE constraints via discovery and write them to CSV, ready "
            "to pass to 'semlog build --declare-file'. Requires the [declare] "
            "extra and the Lydia backend, which is why it is separate from build."
        ),
    )
    mine.add_argument("input", metavar="PATH", help="OCEL file to mine.")
    mine.add_argument(
        "--output",
        metavar="CSV",
        help="Where to write the constraints (default: datasets/declare/<input>.csv).",
    )
    mine.add_argument(
        "--constraints",
        nargs="+",
        metavar="NAME",
        help=(
            "Constraint templates to keep, e.g. Response Precedence Init End. "
            "Omit to keep every template discovery proposes."
        ),
    )
    mine.add_argument(
        "--constraint-support",
        type=float,
        default=0.5,
        metavar="T",
        help="Minimum per-constraint support (default: 0.5).",
    )
    mine.add_argument(
        "--itemset-support",
        type=float,
        default=0.5,
        metavar="T",
        help="Minimum itemset support (default: 0.5).",
    )
    mine.add_argument(
        "--case-concepts",
        nargs="*",
        default=None,
        metavar="TYPE",
        help="Object types to use as cases. Omit to auto-select the most common.",
    )
    mine.add_argument(
        "--auto-case-concepts",
        type=int,
        default=3,
        metavar="N",
        help="How many case concepts to auto-select (default: 3).",
    )
    mine.add_argument(
        "--ocel-extensions",
        nargs="*",
        default=[],
        metavar="EXT",
        help=f"OCEL 2.0 extensions to load: {' | '.join(OCELExtension.spellings())}.",
    )


def _add_reason_parser(sub, parent: argparse.ArgumentParser) -> None:
    """Add the `reason` subcommand and its flags."""
    reason_parser = sub.add_parser(
        "reason",
        parents=[parent],
        help="Materialize or consistency-check an existing ontology.",
        description=(
            "Run a reasoner over an .owl file that has already been built. "
            "Without --output the file is materialized in place."
        ),
    )
    reason_parser.add_argument(
        "onto", metavar="OWL", help="Ontology file to reason over."
    )
    reason_parser.add_argument(
        "--profile",
        default=OWLProfile.OWL2DL.value,
        metavar="PROFILE",
        help=f"OWL profile of the ontology: {' | '.join(OWLProfile.spellings())} (default: OWL2DL).",
    )
    reason_parser.add_argument(
        "--backend",
        type=_reasoner_backend,
        default="HermiT",
        metavar="NAME",
        help=f"Reasoner backend: {' | '.join(REASONER_BACKENDS)} (default: HermiT).",
    )
    reason_parser.add_argument(
        "--output",
        metavar="OWL",
        help="Write the materialized ontology here instead of overwriting the input.",
    )
    reason_parser.add_argument(
        "--check",
        action="store_true",
        help="Only report whether the ontology is consistent, changing nothing.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for the whole command line."""
    parent = _verbosity_parser()
    parser = argparse.ArgumentParser(
        prog="semlog",
        description="Turn OCEL 2.0 event data into OWL 2 ontologies.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"semlog {semlog_version()}"
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    _add_build_parser(sub, parent)
    _add_inspect_parser(sub, parent)
    _add_mine_declare_parser(sub, parent)
    _add_reason_parser(sub, parent)
    return parser


def _log_level(args: argparse.Namespace) -> int:
    """Turn the verbosity flags into a logging level."""
    if getattr(args, "quiet", False):
        return logging.WARNING
    return logging.DEBUG if getattr(args, "verbose", False) else logging.INFO


def build_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Collect the build settings that were actually given on the command line.

    Only arguments that carry a value are kept, so anything left unset falls
    through to the config file. An empty input list is dropped too, since
    positional inputs default to whatever the config names.

    Args:
        args: Parsed arguments.

    Returns:
        Settings to apply on top of the config file.
    """
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key in BUILD_DEFAULTS and value is not None
    }
    if overrides.get("input") == []:
        del overrides["input"]
    return overrides
