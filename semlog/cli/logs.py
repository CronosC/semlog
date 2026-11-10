"""Console and per-input file logging. Start at `setup_console_logging`."""

from __future__ import annotations

import glob
import logging
import os
from datetime import datetime
from pathlib import Path

LOG_KEEP = 10
LOG_FORMAT = "%(asctime)s | %(name)s | %(message)s"
LOG_DATEFMT = "%H:%M:%S"

PACKAGE_LOGGER = "semlog"

_console_handler: logging.Handler | None = None


def _formatter() -> logging.Formatter:
    """Return the format shared by the console and the log files."""
    return logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)


def setup_console_logging(level: int = logging.INFO) -> None:
    """Send this package's log records to the console at `level`.

    The root logger stays at WARNING. A previous console handler is replaced.
    """
    global _console_handler

    root = logging.getLogger()
    if _console_handler is not None:
        root.removeHandler(_console_handler)
        _console_handler.close()

    handler = logging.StreamHandler()
    handler.setFormatter(_formatter())
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    logging.getLogger(PACKAGE_LOGGER).setLevel(level)
    _console_handler = handler


class FileLog:
    """Writes one log file per input and keeps the last `keep` per input.

    With no directory it does nothing. Usable as a context manager.

    Attributes:
        log_dir: Directory for the log files. None disables file logging.
        keep: How many files to keep per input, oldest removed first.
    """

    def __init__(
        self, log_dir: str | os.PathLike[str] | None, keep: int = LOG_KEEP
    ) -> None:
        if keep < 1:
            raise ValueError(f"keep must be at least 1, got {keep}.")
        self.log_dir = Path(log_dir) if log_dir else None
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
        if self.log_dir is None:
            return None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.close()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"{ts}_{stem}.log"
        self._handler = logging.FileHandler(log_path)
        self._handler.setFormatter(_formatter())
        logging.getLogger().addHandler(self._handler)

        # The timestamp width is fixed, so stem `rich` leaves `hiring_rich` alone.
        existing = sorted(self.log_dir.glob(f"????????_??????_{glob.escape(stem)}.log"))
        for old in existing[: max(0, len(existing) - self.keep)]:
            old.unlink()

        return str(log_path)

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
