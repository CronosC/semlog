"""Command line entry point for semlog.

``semlog build`` is the entry point for reproducing a result.  Everything a run
depends on is either a flag, a checked-in TOML config, or recorded in the
manifest written next to the ontology.

    semlog inspect data/ocel/Hiring.xml        # what is in this log?
    semlog build --config runs/hiring.toml     # build it
    semlog reason data/onto/hiring.owl         # materialise it

Argument parsing lives here and nowhere else: flags are resolved into a
:class:`~semanticlog.cli.config.RunConfig` and the pipeline in
:mod:`semanticlog.cli.run` takes it from there.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
import textwrap
from pathlib import Path

from semanticlog.cli.config import (
    BUILD_DEFAULTS,
    ConfigError,
    describe,
    enum_choices,
    parse_enum,
    resolve_run_config,
)
from semanticlog.cli.logs import setup_console_logging
from semanticlog.cli.manifest import semlog_version
from semanticlog.cli.run import (
    inspect_input,
    mine_declare,
    reason,
    run_build,
)
from semanticlog.onto.settings import (
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OcelExtension,
    OwlProfile,
)

logger = logging.getLogger("semlog.cli")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

_REASONER_BACKENDS = ["HermiT", "Pellet", "ELK"]

_EPILOG = textwrap.dedent("""\
    examples:
      semlog inspect data/ocel/Hiring.xml
      semlog build data/ocel/Hiring.xml --case-concepts Candidate Recruiter
      semlog build --config runs/hiring.toml
      semlog build --config runs/hiring.toml --dry-run
      semlog mine-declare data/ocel/enron.json --constraints Response Precedence
      semlog reason data/onto/hiring.owl --profile OWL2EL

    every build writes <name>.owl and <name>.manifest.json side by side; the
    manifest records the input checksum, the resolved settings and the semlog
    version, so a result can be traced back to the run that produced it.
    """)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _verbosity_parser() -> argparse.ArgumentParser:
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
        help=f"Directory for the ontology and manifest (default: {BUILD_DEFAULTS['output']}).",
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
        choices=enum_choices(OwlProfile),
        metavar="PROFILE",
        help=f"OWL profile: {' | '.join(enum_choices(OwlProfile))} (default: {BUILD_DEFAULTS['profile']}).",
    )
    build.add_argument(
        "--order",
        nargs="*",
        metavar="MODE",
        help=(
            "Event order representations: "
            f"{' | '.join(enum_choices(EventOrderRepresentation))}. "
            "Pass --order with no values for none. Default: df."
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
        "--data-representation",
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
        help="Bin labels, exactly --bins of them (default: very_low low average high very_high).",
    )
    build.add_argument(
        "--inverse-representation",
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
            f"OCEL 2.0 extensions: {' | '.join(enum_choices(OcelExtension))}. "
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
            "Required when --order includes declare."
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
        choices=_REASONER_BACKENDS,
        metavar="NAME",
        help=f"Reasoner used by --materialize: {' | '.join(_REASONER_BACKENDS)} (default: HermiT).",
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
        help="Also write the loaded OCEDStore as a pickle.",
    )
    build.add_argument(
        "--size-limit",
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
    inspect = sub.add_parser(
        "inspect",
        parents=[parent],
        help="Summarise an OCEL file without building anything.",
        description=(
            "Print the activities, object types, relations and time range of an "
            "OCEL file, plus the case concepts a build would auto-select. Names "
            "are shown in the cleaned form that --case-concepts expects."
        ),
    )
    inspect.add_argument(
        "input", nargs="+", metavar="PATH", help="OCEL files to summarise."
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
        help=f"OCEL 2.0 extensions to load: {' | '.join(enum_choices(OcelExtension))}.",
    )


def _add_mine_declare_parser(sub, parent: argparse.ArgumentParser) -> None:
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
        help="Where to write the constraints (default: data/declare/<input>.csv).",
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
        help=f"OCEL 2.0 extensions to load: {' | '.join(enum_choices(OcelExtension))}.",
    )


def _add_reason_parser(sub, parent: argparse.ArgumentParser) -> None:
    reason_parser = sub.add_parser(
        "reason",
        parents=[parent],
        help="Materialise or consistency-check an existing ontology.",
        description=(
            "Run a reasoner over an .owl file that has already been built. "
            "Without --output the file is materialised in place."
        ),
    )
    reason_parser.add_argument(
        "onto", metavar="OWL", help="Ontology file to reason over."
    )
    reason_parser.add_argument(
        "--profile",
        choices=enum_choices(OwlProfile),
        default=OwlProfile.OWL2DL.value,
        metavar="PROFILE",
        help=f"OWL profile of the ontology: {' | '.join(enum_choices(OwlProfile))} (default: OWL2DL).",
    )
    reason_parser.add_argument(
        "--backend",
        choices=_REASONER_BACKENDS,
        default="HermiT",
        metavar="NAME",
        help=f"Reasoner backend: {' | '.join(_REASONER_BACKENDS)} (default: HermiT).",
    )
    reason_parser.add_argument(
        "--output",
        metavar="OWL",
        help="Write the materialised ontology here instead of overwriting the input.",
    )
    reason_parser.add_argument(
        "--check",
        action="store_true",
        help="Only report whether the ontology is consistent, changing nothing.",
    )


def build_parser() -> argparse.ArgumentParser:
    """The full argument parser, including every subcommand."""
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


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _log_level(args: argparse.Namespace) -> int:
    if getattr(args, "quiet", False):
        return logging.WARNING
    return logging.DEBUG if getattr(args, "verbose", False) else logging.INFO


def build_overrides(args: argparse.Namespace) -> dict[str, object]:
    """The settings the user gave explicitly, keyed as the config file keys are.

    Any parser destination that matches a config key and is not ``None``
    counts as given, which is why every build flag defaults to ``None``.
    ``input`` is the exception: argparse cannot tell an absent ``nargs="*"``
    positional from an empty one, so an empty list means "fall back to the
    config" rather than "build nothing".
    """
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key in BUILD_DEFAULTS and value is not None
    }
    if overrides.get("input") == []:
        del overrides["input"]
    return overrides


def _command_line(argv: list[str]) -> str:
    return "semlog " + shlex.join(argv)


def _cmd_build(args: argparse.Namespace, argv: list[str]) -> int:
    cfg = resolve_run_config(
        build_overrides(args),
        config_file=args.config,
        command=_command_line(argv),
    )

    if args.dry_run:
        print(describe(cfg))
        return EXIT_OK

    report = run_build(cfg)
    logger.info(f"Done: {report.summary()}")
    for path, message in report.failed:
        logger.error(f"{path}: {message}")
    return EXIT_OK if report.ok else EXIT_FAILED


def _cmd_inspect(args: argparse.Namespace, argv: list[str]) -> int:
    extensions = [
        parse_enum(OcelExtension, token, key="ocel_extensions")
        for token in args.ocel_extensions
    ]
    reports = [
        inspect_input(Path(path), extensions, suggest=args.suggest)
        for path in args.input
    ]
    print("\n\n".join(report.render() for report in reports))
    return EXIT_OK


def _cmd_mine_declare(args: argparse.Namespace, argv: list[str]) -> int:
    input_path = Path(args.input)
    output = (
        Path(args.output)
        if args.output
        else Path("data/declare") / f"{input_path.stem}.csv"
    )
    extensions = [
        parse_enum(OcelExtension, token, key="ocel_extensions")
        for token in args.ocel_extensions
    ]
    mine_declare(
        input_path=input_path,
        output_path=output,
        case_concepts=args.case_concepts,
        constraints=args.constraints,
        constraint_support=args.constraint_support,
        itemset_support=args.itemset_support,
        auto_case_concepts=args.auto_case_concepts,
        extensions=extensions,
    )
    return EXIT_OK


def _cmd_reason(args: argparse.Namespace, argv: list[str]) -> int:
    consistent = reason(
        onto_path=Path(args.onto),
        profile=parse_enum(OwlProfile, args.profile, key="profile"),
        backend=args.backend,
        output_path=Path(args.output) if args.output else None,
        check_only=args.check,
    )
    return EXIT_OK if consistent else EXIT_FAILED


_COMMANDS = {
    "build": _cmd_build,
    "inspect": _cmd_inspect,
    "mine-declare": _cmd_mine_declare,
    "reason": _cmd_reason,
}


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Arguments without the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` when a build failed or an ontology was
        inconsistent, ``2`` for an unusable configuration.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_USAGE

    setup_console_logging(_log_level(args))
    try:
        return _COMMANDS[args.command](args, argv)
    except ConfigError as exc:
        print(f"semlog {args.command}: error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except FileNotFoundError as exc:
        print(f"semlog {args.command}: error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
