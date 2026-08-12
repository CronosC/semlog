"""The `semlog` command line.

Flags and TOML configs are resolved into one `RunConfig` first, and the
subcommands then work from that value. This is the only layer that knows about
argparse.
"""

from semanticlog.cli.main import EXIT_FAILED, EXIT_OK, EXIT_USAGE, main
from semanticlog.cli.parser import build_parser

__all__ = ["EXIT_FAILED", "EXIT_OK", "EXIT_USAGE", "build_parser", "main"]
