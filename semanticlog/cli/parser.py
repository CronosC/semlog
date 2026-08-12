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
from importlib.metadata import PackageNotFoundError, version

from semanticlog.cli.config import BUILD_DEFAULTS, enum_choices
from semanticlog.onto.settings import (
    DEFAULT_BIN_LABELS,
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OCELExtension,
    OWLProfile,
)

_REASONER_BACKENDS = ["HermiT", "Pellet", "ELK"]


def semlog_version() -> str:
    """Return the installed version, or `unknown` when running from a checkout."""
    try:
        return version("semlog")
    except PackageNotFoundError:
        return "unknown"


def _reasoner_backend(value: str) -> str:
    """Accept a backend name in any case, and reject an unknown one."""
    for known in _REASONER_BACKENDS:
        if value.casefold() == known.casefold():
            return known
    raise argparse.ArgumentTypeError(
        f"unknown reasoner backend {value!r}; choose from "
        + ", ".join(_REASONER_BACKENDS)
    )


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
    build.add_argument(
        "--output",
        metavar="DIR",
        help=f"Directory for the ontology (default: {BUILD_DEFAULTS['output']}).",
    )
    build.add_argument(
        "--name",
        metavar="STEM",
        help=(
            "Shared stem for every artifact of this run. Defaults to the "
            "--config file name for a single input, otherwise to the input name."
        ),
    )
    build.add_argument(
        "--profile",
        metavar="PROFILE",
        help=f"OWL profile: {' | '.join(enum_choices(OWLProfile))} (default: {BUILD_DEFAULTS['profile']}).",
    )
    build.add_argument(
        "--order-representations",
        nargs="*",
        metavar="MODE",
        help=(
            "Event order representations: "
            f"{' | '.join(enum_choices(EventOrderRepresentation))}. "
            "Pass --order-representations with no values for none. Default: df."
        ),
    )
    build.add_argument(
        "--case-concepts",
        nargs="*",
        metavar="TYPE",
        help=(
            "Object types to use as cases. Omit to auto-select the most common "
            "types; pass --case-concepts with no values for none. Run "
            "'semlog inspect' to see the available names."
        ),
    )
    build.add_argument(
        "--auto-case-concepts",
        type=int,
        metavar="N",
        help=f"How many object types to auto-select (default: {BUILD_DEFAULTS['auto_case_concepts']}).",
    )
    build.add_argument(
        "--data-representations",
        nargs="*",
        metavar="MODE",
        help=(
            "How numeric attributes are represented: "
            f"{' | '.join(enum_choices(DataRepresentation))}. "
            "values keeps the raw number, equal_bins buckets by frequency, "
            "dist_bins fits a distribution first. equal_bins and dist_bins are "
            "mutually exclusive. Pass no values to omit numeric attributes. "
            "Default: values."
        ),
    )
    build.add_argument(
        "--bins",
        type=int,
        metavar="N",
        help=f"Bin count for the binning modes (default: {BUILD_DEFAULTS['bins']}).",
    )
    build.add_argument(
        "--bin-labels",
        nargs="+",
        metavar="LABEL",
        help=(
            "Bin labels, exactly --bins of them "
            f"(default: {' '.join(DEFAULT_BIN_LABELS)})."
        ),
    )
    build.add_argument(
        "--inverse-representations",
        nargs="*",
        metavar="MODE",
        help=(
            "Inverse relation strategies: "
            f"{' | '.join(enum_choices(InverseRepresentation))}. "
            "axiomatic adds InverseObjectProperty axioms (OWL2DL only), "
            "implicit materialises the inverse assertions. Default: none."
        ),
    )
    build.add_argument(
        "--ocel-extensions",
        nargs="*",
        metavar="EXT",
        help=(
            f"OCEL 2.0 extensions: {' | '.join(enum_choices(OCELExtension))}. "
            "Default: none."
        ),
    )
    build.add_argument(
        "--annotation-document",
        metavar="JSON",
        help="JSON file mapping entity names to {label, comment} dicts.",
    )
    build.add_argument(
        "--derive-annotations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Emit rdfs:label on individuals derived from the source data (default: off).",
    )
    build.add_argument(
        "--declare-file",
        metavar="CSV",
        help=(
            "DECLARE constraints CSV, as written by 'semlog mine-declare'. "
            "Required when --order-representations includes declare."
        ),
    )
    build.add_argument(
        "--materialize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run the reasoner over each ontology after writing it (default: off). "
            "'semlog reason' does the same thing to an existing file."
        ),
    )
    build.add_argument(
        "--reasoner-backend",
        type=_reasoner_backend,
        metavar="NAME",
        help=f"Reasoner used by --materialize: {' | '.join(_REASONER_BACKENDS)} (default: HermiT).",
    )
    build.add_argument(
        "--streaming",
        action="store_true",
        default=None,
        help=(
            "Write OWL functional syntax straight to disk at near-constant "
            "memory, for logs too large to hold. No ontology is kept, so "
            "--materialize is unavailable."
        ),
    )
    build.add_argument(
        "--save-backmap",
        action="store_true",
        default=None,
        help="Also write the cleaned-name to original-name map as JSON.",
    )
    build.add_argument(
        "--save-store",
        action="store_true",
        default=None,
        help="Also write the loaded OCEDStore as a Parquet directory.",
    )
    build.add_argument(
        "--size-limit-mb",
        type=int,
        metavar="MB",
        help="Skip inputs larger than this many MB (default: 0, no limit).",
    )
    build.add_argument(
        "--log-dir",
        metavar="DIR",
        help="Directory for per-input log files (default: no file logging).",
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
        help=f"OCEL 2.0 extensions to load: {' | '.join(enum_choices(OCELExtension))}.",
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
        help="Constraint templates to mine, e.g. Response Precedence Init End.",
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
        help=f"OCEL 2.0 extensions to load: {' | '.join(enum_choices(OCELExtension))}.",
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
        help=f"OWL profile of the ontology: {' | '.join(enum_choices(OWLProfile))} (default: OWL2DL).",
    )
    reason_parser.add_argument(
        "--backend",
        type=_reasoner_backend,
        default="HermiT",
        metavar="NAME",
        help=f"Reasoner backend: {' | '.join(_REASONER_BACKENDS)} (default: HermiT).",
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
