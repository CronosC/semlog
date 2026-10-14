"""Results of a pre-build check, and how several are raised together.

A check returns a `CheckReport`: a header line, one line per finding, and
advice on what to change. `raise_reports` collects the reports of every check
and raises one error, so one run reports everything wrong with an input.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from semanticlog.core.errors import ConfigError, InputError


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
        header: The line printed above the findings.
        problems: One line per finding, each indented when the report is
            rendered. Empty when the header says everything.
        advice: What to change, written after the findings.
        about: Where the fix is made, the config or the input.
    """

    header: str
    problems: tuple[str, ...] = ()
    advice: str = ""
    about: Fixed = Fixed.INPUT

    def render(self) -> str:
        """Return the block this report contributes to a message."""
        block = self.header
        if self.problems:
            block += "\n" + "\n".join(f"  {problem}" for problem in self.problems)
        if self.advice:
            block += "\n\n" + self.advice
        return block


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
    message = "\n\n".join(report.render() for report in found)
    if all(report.about is Fixed.CONFIG for report in found):
        raise ConfigError(message)
    raise InputError(message)
