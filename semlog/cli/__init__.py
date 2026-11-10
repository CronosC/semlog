"""The `semlog` command line. Start at `main`.

Flags and a TOML config resolve to a `RunConfig` for `build` and a `LogInputs`
for `inspect` and `mine-declare`.
"""

from semlog.cli.main import (
    EXIT_FAILED,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_USAGE,
    main,
)
from semlog.cli.parser import build_parser

__all__ = [
    "EXIT_FAILED",
    "EXIT_INTERRUPTED",
    "EXIT_OK",
    "EXIT_USAGE",
    "build_parser",
    "main",
]
