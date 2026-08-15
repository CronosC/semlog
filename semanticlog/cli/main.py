"""Subcommand dispatch and exit codes."""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from pathlib import Path

from semanticlog.cli.config import (
    parse_enum,
    resolve_run_config,
)
from semanticlog.cli.logs import setup_console_logging
from semanticlog.cli.parser import (
    _column_overrides,
    _log_level,
    build_overrides,
    build_parser,
)
from semanticlog.cli.run import (
    build_all,
    inspect_input,
    mine_declare,
    reason,
)
from semanticlog.core.enums import (
    OCELExtension,
    OWLProfile,
)
from semanticlog.core.errors import SemlogError

logger = logging.getLogger("semlog.cli")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _command_line(argv: list[str]) -> str:
    """Rebuild the command as typed, for the provenance record."""
    return "semlog " + shlex.join(argv)


def _cmd_build(args: argparse.Namespace) -> int:
    """Resolve the configuration and build, or print it under --dry-run."""
    cfg = resolve_run_config(
        build_overrides(args),
        config_file=args.config,
        command=args.command_line,
    )

    if args.dry_run:
        print(cfg.render())
        return EXIT_OK

    report = build_all(cfg)
    logger.info(f"Done: {report.summary()}")
    for path, message in report.failed:
        logger.error(f"{path}: {message}")
    return EXIT_OK if report.ok else EXIT_FAILED


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Summarize each input log."""
    extensions = [
        parse_enum(OCELExtension, token, key="ocel_extensions")
        for token in args.ocel_extensions
    ]
    columns = _column_overrides(args)
    reports = [
        inspect_input(Path(path), extensions, suggest=args.suggest, columns=columns)
        for path in args.input
    ]
    print("\n\n".join(report.render() for report in reports))
    return EXIT_OK


def _cmd_mine_declare(args: argparse.Namespace) -> int:
    """Mine DECLARE constraints into a CSV."""
    input_path = Path(args.input)
    output = (
        Path(args.output)
        if args.output
        else Path("datasets/declare") / f"{input_path.stem}.csv"
    )
    extensions = [
        parse_enum(OCELExtension, token, key="ocel_extensions")
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
        columns=_column_overrides(args),
    )
    return EXIT_OK


def _cmd_reason(args: argparse.Namespace) -> int:
    """Materialize or consistency-check an existing ontology."""
    consistent = reason(
        onto_path=Path(args.onto),
        profile=parse_enum(OWLProfile, args.profile, key="profile"),
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
    """Run one semlog command.

    A `SemlogError` is a problem the caller can fix, and is reported as a
    message: a bad setting, a log that cannot be read, a solver that is not
    installed. Any other exception propagates with its traceback, since it
    indicates a defect rather than something to fix in the input.

    Args:
        argv: Arguments without the program name. Defaults to the real ones.

    Returns:
        0 on success, 1 if a build failed or an ontology was inconsistent,
        2 if the command or its configuration was unusable.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_USAGE

    args.command_line = _command_line(argv)

    setup_console_logging(_log_level(args))
    try:
        return _COMMANDS[args.command](args)
    except (SemlogError, FileNotFoundError) as exc:
        print(f"semlog {args.command}: error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
