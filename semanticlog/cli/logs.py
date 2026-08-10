"""Console and per-input file logging for the CLI.

Only :func:`setup_console_logging` touches the root logger, and it replaces just
the handler it installed itself, so importing the CLI from a notebook or a test
does not discard the host's own logging configuration.
"""

from __future__ import annotations

import glob
import logging
import os
from datetime import datetime

LOG_KEEP = 10
LOG_FORMAT = "%(name)s | %(asctime)s | %(message)s"
LOG_DATEFMT = "%H:%M:%S"

# Loggers that are informative for us but noisy for a reader of the console.
_QUIET_LIBRARIES = ("pm4py", "matplotlib", "numexpr")

_console_handler: logging.Handler | None = None


def _formatter() -> logging.Formatter:
    return logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)


def setup_console_logging(level: int = logging.INFO) -> None:
    """Send semlog's log records to stderr at ``level``.

    Calling this more than once replaces the previous console handler rather
    than stacking a second one.

    Args:
        level: Root log level to install, e.g. ``logging.DEBUG``.
    """
    global _console_handler

    root = logging.getLogger()
    if _console_handler is not None:
        root.removeHandler(_console_handler)
        _console_handler.close()

    handler = logging.StreamHandler()
    handler.setFormatter(_formatter())
    root.addHandler(handler)
    root.setLevel(level)
    _console_handler = handler

    for name in _QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))


class FileLog:
    """Owns the per-input file log handler, so that a run holds no module state.

    Args:
        log_dir: Directory to write logs into. ``None`` disables file logging.
        keep: How many previous logs to retain per input stem.
    """

    def __init__(self, log_dir: str | os.PathLike[str] | None, keep: int = LOG_KEEP):
        self.log_dir = str(log_dir) if log_dir else ""
        self.keep = keep
        self._handler: logging.FileHandler | None = None

    def switch_to(self, stem: str) -> str | None:
        """Route logging to a fresh file for ``stem`` and prune old ones.

        Returns:
            The new log file path, or ``None`` when file logging is disabled.
        """
        if not self.log_dir:
            return None
        os.makedirs(self.log_dir, exist_ok=True)
        self.close()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.log_dir, f"{ts}_{stem}.log")
        self._handler = logging.FileHandler(log_path)
        self._handler.setFormatter(_formatter())
        logging.getLogger().addHandler(self._handler)

        existing = sorted(glob.glob(os.path.join(self.log_dir, f"*_{stem}.log")))
        for old in existing[: -self.keep]:
            os.remove(old)

        return log_path

    def close(self) -> None:
        """Detach and close the current handler, if any."""
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            self._handler.close()
            self._handler = None

    def __enter__(self) -> FileLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
