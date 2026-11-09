"""Results of a pre-build check, and how several are raised together.

A check returns a `CheckReport`: a header, the findings, and advice on what to
change. `raise_reports` collects the reports of every check and raises one
error of one line, so one run reports everything wrong with an input.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from semanticlog.core.errors import ConfigError, InputError

_SHOWN = 5


def first_five(items: Sequence[object]) -> str:
    """Join the first five items with commas, counting the rest as `and N more`."""
    shown = ", ".join(str(item) for item in items[:_SHOWN])
    more = len(items) - _SHOWN
    return f"{shown} and {more} more" if more > 0 else shown


class Fixed(str, Enum):
    """Where a finding is fixed.

    Attributes:
        CONFIG: In the build configuration, reported as a `ConfigError`.
        INPUT: In the log or another file the build reads, reported as an
            `InputError`.
    """

    CONFIG = "config"
    INPUT = "input"


@dataclass(frozen=True)
class CheckReport:
    """The findings of one check, with the header and advice around them.

    Attributes:
        header: The sentence before the findings.
        problems: One clause per finding. The first five are rendered and
            the rest counted. Empty when the header says everything.
        advice: What to change, written after the findings.
        about: Where the fix is made, the config or the input.
    """

    header: str
    problems: tuple[str, ...] = ()
    advice: str = ""
    about: Fixed = Fixed.INPUT

    def render(self) -> str:
        """Return the one line this report contributes to a message."""
        parts = [self.header]
        shown = [problem.removesuffix(".") for problem in self.problems[:_SHOWN]]
        more = len(self.problems) - _SHOWN
        if more > 0:
            shown.append(f"{more} more not listed")
        parts += [f"{problem}." for problem in shown]
        if self.advice:
            parts.append(self.advice)
        return " ".join(parts)


def raise_reports(reports: Iterable[CheckReport | None]) -> None:
    """Raise one error carrying everything the checks found.

    Args:
        reports: What each check returned, in the order they ran. None for a
            check that found nothing or did not apply.

    Raises:
        ConfigError: If every report is about the configuration.
        InputError: If any report is about an input.
    """
    found = [report for report in reports if report is not None]
    if not found:
        return
    message = " ".join(report.render() for report in found)
    if all(report.about is Fixed.CONFIG for report in found):
        raise ConfigError(message)
    raise InputError(message)
