"""Subcommand dispatch and exit codes.

Results go to stdout: the paths a build wrote, the `inspect` report, the
`reason --check` result. Log records go to stderr. `-q` lowers the log level
and leaves stdout alone.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from semanticlog.cli.config import (
    resolve_log_inputs,
    resolve_run_config,
)
from semanticlog.cli.logs import setup_console_logging
from semanticlog.cli.parser import (
    _log_level,
    build_parser,
    flag_overrides,
)
from semanticlog.cli.settings_table import setting
from semanticlog.core.errors import SemlogError

# The command bodies are imported inside each command, so that `--help` and
# argument errors do not load the builder, the loaders or the OWL API.

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def _cmd_build(args: argparse.Namespace) -> int:
    """Resolve the configuration and build, or print it under --dry-run."""
    cfg = resolve_run_config(
        flag_overrides(args),
        config_file=args.config,
    )

    if args.dry_run:
        for message in cfg.warnings:
            logger.warning(message)
        print(cfg.render())
        return EXIT_OK

    from semanticlog.cli.run import build_all

    report = build_all(cfg)
    for input_path in report.built:
        print(cfg.ontology_path(input_path))
        for path in cfg.converted_paths(input_path).values():
            print(path)
    logger.info("Done: %s", report.summary())
    return EXIT_OK if report.ok else EXIT_FAILED


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Summarize each input log, printing each one as it is read.

    A log that cannot be read is logged and the rest are summarized.
    """
    from semanticlog.cli.inspect import inspect_input

    inputs = resolve_log_inputs(flag_overrides(args), config_file=args.config)
    for message in inputs.warnings:
        logger.warning(message)

    printed, failed = 0, 0
    for path in inputs.inputs:
        try:
            report = inspect_input(
                path,
                inputs.ocel_extensions,
                columns=inputs.columns,
            )
        except (SemlogError, OSError) as exc:
            failed += 1
            if str(path) in str(exc):
                logger.error("%s", exc)
            else:
                logger.error("Failed while reading %s: %s", path.name, exc)
            continue
        except Exception as exc:
            failed += 1
            logger.exception("Failed while reading %s: %s", path.name, exc)
            continue
        if printed:
            print()
        print(report.render())
        printed += 1
    return EXIT_OK if not failed else EXIT_FAILED


def _cmd_mine_declare(args: argparse.Namespace) -> int:
    """Mine DECLARE constraints into a JSON file.

    Without a declare_file the file goes to `<input stem>.declare.json` in the
    current directory.
    """
    inputs = resolve_log_inputs(flag_overrides(args), config_file=args.config)
    for message in inputs.warnings:
        logger.warning(message)

    from semanticlog.cli.run import mine_declare

    output = inputs.declare_file
    written = mine_declare(
        input_path=inputs.one_input(),
        output_path=Path(output) if output else None,
        case_concepts=inputs.case_concepts,
        constraints=inputs.constraints,
        constraint_support=inputs.constraint_support,
        itemset_support=inputs.itemset_support,
        auto_case_concepts=inputs.auto_case_concepts,
        extensions=inputs.ocel_extensions,
        columns=inputs.columns,
    )
    print(written)
    return EXIT_OK


def _cmd_reason(args: argparse.Namespace) -> int:
    """Materialize or consistency-check an existing ontology."""
    from semanticlog.cli.run import reason

    onto_path = Path(args.onto)
    output_path = Path(args.output) if args.output else None
    consistent = reason(
        onto_path=onto_path,
        backend=setting("reasoner_backend").read(
            args.reasoner_backend, "--reasoner-backend"
        ),
        output_path=output_path,
        check_only=args.check,
    )
    if args.check:
        print(f"{onto_path}: {'consistent' if consistent else 'INCONSISTENT'}")
        return EXIT_OK if consistent else EXIT_FAILED
    print(output_path or onto_path)
    return EXIT_OK


_COMMANDS = {
    "build": _cmd_build,
    "inspect": _cmd_inspect,
    "mine-declare": _cmd_mine_declare,
    "reason": _cmd_reason,
}


def main(argv: list[str] | None = None) -> int:
    """Run one semlog command and return its exit code.

    A `SemlogError` or `OSError` that reaches this function is printed as one
    line. Any other exception propagates with its traceback.

    Args:
        argv: Arguments without the program name, or None for `sys.argv`.

    Returns:
        0 on success. 1 if a build failed, `inspect` could not read a log, or
        an ontology was inconsistent. 2 if no command was given or the command
        raised a `SemlogError` or `OSError`, such as an unusable configuration
        or a log `mine-declare` could not read. 130 if it was interrupted.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_USAGE

    try:
        setup_console_logging(_log_level(args))
        return _COMMANDS[args.command](args)
    except KeyboardInterrupt:
        print(f"semlog {args.command}: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (SemlogError, OSError) as exc:
        print(f"semlog {args.command}: error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
