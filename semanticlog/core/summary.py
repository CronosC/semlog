"""Formatting for the lines a build step logs about what it produced.

A step reports a count line and, under it, the names behind those counts
grouped by kind. Each line returned here is one log record, because the
console format prefixes every record and a continuation line inside one
record would start at column zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

LINE_WIDTH = 80
LABEL_WIDTH = 19
INDENT = "    "
MAX_LINES = 3


def count_phrase(
    counts: Iterable[tuple[int, str, str]], *, keep_zero: bool = False
) -> str:
    """Join counts into a phrase, leaving out the ones that are zero.

    Args:
        counts: Triples of a count, the word for one of them and the word for
            several.
        keep_zero: Keep a count of zero, for a phrase that has to name every
            part of what it measured.

    Returns:
        A phrase such as `7 classes, 1 data property`, or an empty string when
        every count is zero and they are not kept.
    """
    return ", ".join(
        f"{count:,} {singular if count == 1 else plural}"
        for count, singular, plural in counts
        if count or keep_zero
    )


def name_lines(
    label: str, names: Sequence[str], *, complete: bool = False
) -> list[str]:
    """Lay out names under a label, one list per line.

    Args:
        label: What these names are, written above them in its own column.
        names: The names, in the order to print them.
        complete: Print all of them. Otherwise the list stops after
            `MAX_LINES` and says how many are left.

    Returns:
        The lines to log, already indented. Empty when there are no names.
    """
    if not names:
        return []

    head = INDENT + f"{label}:".ljust(LABEL_WIDTH)
    hang = " " * len(head)
    packed = _packed(names, LINE_WIDTH - len(head))

    if complete or len(packed) <= MAX_LINES:
        rendered = [", ".join(group) for group in packed]
    else:
        shown = packed[:MAX_LINES]
        dropped = sum(len(group) for group in packed[MAX_LINES:])
        rendered = [", ".join(group) for group in shown[:-1]]
        rendered.append(_cut(shown[-1], dropped, len(hang)))

    lines = [f"{group}," for group in rendered[:-1]] + rendered[-1:]
    return [(head if index == 0 else hang) + line for index, line in enumerate(lines)]


def _packed(names: Sequence[str], width: int) -> list[list[str]]:
    """Group names into lines of `width`, filling each before starting the next.

    A name is never split, since one can hold a space of its own, such as an
    attribute listed with the types that share it.
    """
    lines: list[list[str]] = [[]]
    for index, name in enumerate(names):
        candidate = [*lines[-1], name]
        # Every line but the last ends on a comma, so leave room for it.
        length = len(", ".join(candidate)) + (index < len(names) - 1)
        if lines[-1] and length > width:
            lines.append([name])
        else:
            lines[-1] = candidate
    return lines


def _cut(group: Sequence[str], dropped: int, start: int) -> str:
    """Render the last line of a list, saying how many names it leaves out.

    Names are given up from the end of the line until the count of what is
    left out fits beside the ones that stay.
    """
    kept = list(group)
    while True:
        marker = f", ... (+{dropped:,} more)"
        text = ", ".join(kept)
        if len(kept) <= 1 or start + len(text) + len(marker) <= LINE_WIDTH:
            return text + marker
        kept.pop()
        dropped += 1
