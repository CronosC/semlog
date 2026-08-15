"""The failures semlog reports to whoever ran it.

Every one of these is something the caller can fix by changing a setting, a
log, or what is installed. They subclass `ValueError`, because each of them is
a value that will not do, and the CLI catches `SemlogError` rather than
`ValueError`.

That is the line: a plain `ValueError` from inside the package means the code
asked for something impossible, such as a builder used twice or a sink written
to after it closed. Those keep their traceback rather than being printed as a
message about the input.
"""

from __future__ import annotations


class SemlogError(ValueError):
    """A problem the caller can fix. The base of everything below."""


class ConfigError(SemlogError):
    """A setting is missing, unknown, of the wrong sort, or contradicts another."""


class InputError(SemlogError):
    """A log or a data file is not something a build can use."""


class DependencyError(SemlogError):
    """Something semlog runs on is not installed, such as a solver or a JVM."""
