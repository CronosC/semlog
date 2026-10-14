"""Errors the caller can fix by changing a setting, a log or an installed dependency.

All of them subclass `ValueError`. The CLI catches `SemlogError` and prints its
message. A plain `ValueError` from inside the package marks a programming
error, such as a builder used twice or a sink written to after `close()`, and
keeps its traceback.
"""

from __future__ import annotations


class SemlogError(ValueError):
    """Base of every error the caller can fix."""


class ConfigError(SemlogError):
    """A setting is missing, unknown, of the wrong sort, or contradicts another."""


class InputError(SemlogError):
    """A log or a data file is not something a build can use."""


class DependencyError(SemlogError):
    """A dependency such as a solver or a JVM is not installed."""
