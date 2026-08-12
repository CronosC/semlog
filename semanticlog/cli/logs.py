"""Logging setup, which only the CLI does.

The library logs through module loggers and configures nothing. Handlers are
attached here, so importing semlog never changes how the calling program logs.
"""

from __future__ import annotations

import glob
import logging
import os
from datetime import datetime

LOG_KEEP = 10
LOG_FORMAT = "%(name)s | %(asctime)s | %(message)s"
LOG_DATEFMT = "%H:%M:%S"

_QUIET_LIBRARIES = ("pm4py", "matplotlib", "numexpr")

_console_handler: logging.Handler | None = None


def _formatter() -> logging.Formatter:
    """Return the format shared by the console and the log files."""
    return logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)


def setup_console_logging(level: int = logging.INFO) -> None:
    """Send log records to the console at the given level.

    A previous console handler is removed first, so calling this twice does
    not double every line. Noisy libraries are held at warning level whatever
    the level asked for.
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
    """Writes a separate log file per input, and keeps the last few.

    Does nothing at all when no directory was configured, so a build can be
    wrapped in it either way. Can be used as a context manager.
    """

    def __init__(self, log_dir: str | os.PathLike[str] | None, keep: int = LOG_KEEP):
        """Prepare file logging.

        Args:
            log_dir: Directory for the log files. None disables file logging.
            keep: How many files to keep per input, oldest removed first.
        """
        self.log_dir = str(log_dir) if log_dir else ""
        self.keep = keep
        self._handler: logging.FileHandler | None = None

    def switch_to(self, stem: str) -> str | None:
        """Start a new log file for one input, closing the previous one.

        Args:
            stem: Output name of the input, used in the file name along with
                a timestamp.

        Returns:
            The file being written, or None when file logging is off.
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
        """Stop writing to the current file, if there is one."""
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            self._handler.close()
            self._handler = None

    def __enter__(self) -> FileLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
