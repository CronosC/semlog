"""Log lines that report what a build step produced.

A step logs a count line and, under it, the names behind the counts grouped
by kind. `report_by_kind` writes both. Every line is one log record, because
the console format prefixes each record.

A step passes its own logger, so the record names the step's module.
`MAX_LINES` caps how much of a long list is printed at INFO level.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence

LINE_WIDTH = 80
LABEL_WIDTH = 19
INDENT = "    "
MAX_LINES = 3

COUNT_INDENT = "  "


def count_phrase(
    counts: Iterable[tuple[int, str, str]], *, keep_zero: bool = False
) -> str:
    """Join counts into a phrase, leaving out the ones that are zero.

    Args:
        counts: Triples of a count, the word for one of them and the word for
            several.
        keep_zero: Keep counts of zero.

    Returns:
        A phrase such as `7 classes, 1 data property`, or an empty string when
        every count is zero and they are not kept.
    """
    return ", ".join(
        f"{count:,} {singular if count == 1 else plural}"
        for count, singular, plural in counts
        if count or keep_zero
    )


def counted(count: int, singular: str, plural: str) -> str:
    """Return one count with its word, such as `1 union`, or "" for zero."""
    return count_phrase([(count, singular, plural)])


def name_lines(
    label: str, names: Sequence[str], *, complete: bool = False
) -> list[str]:
    """Return the indented lines that list names under a label.

    Args:
        complete: Print every name. Otherwise the list stops after `MAX_LINES`
            and says how many are left.
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
    """Group names into lines of at most `width` characters, never splitting a name."""
    lines: list[list[str]] = [[]]
    for index, name in enumerate(names):
        candidate = [*lines[-1], name]
        # Every line but the last ends on a comma, so leave room for it.
        length = len(", ".join(candidate)) + (1 if index < len(names) - 1 else 0)
        if lines[-1] and length > width:
            lines.append([name])
        else:
            lines[-1] = candidate
    return lines


def _cut(group: Sequence[str], dropped: int, start: int) -> str:
    """Render the last line of a cut list, ending on a count of the omitted names."""
    kept = list(group)
    while kept:
        marker = f", ... (+{dropped:,} more)"
        text = ", ".join(kept)
        if start + len(text) + len(marker) <= LINE_WIDTH:
            return text + marker
        kept.pop()
        dropped += 1
    return f"... (+{dropped:,} more)"


def report_names(
    logger: logging.Logger,
    label: str,
    names: Sequence[str],
    *,
    level: int = logging.INFO,
) -> None:
    """Log the names behind a count, under their label.

    The list is cut after `MAX_LINES` unless the logger is enabled for DEBUG.
    """
    complete = logger.isEnabledFor(logging.DEBUG)
    for line in name_lines(label, names, complete=complete):
        logger.log(level, line)


def report_by_kind(
    logger: logging.Logger,
    words: Mapping[str, tuple[str, str]],
    names: Mapping[str, Sequence[str]],
    *,
    lead: str = "",
    extra: Iterable[tuple[int, str, str]] = (),
) -> None:
    """Log what a step produced, as a count line and then the names under it.

    A kind with no names is left out of both. When every count is zero,
    nothing is logged. A kind in `names` that `words` does not list is logged
    at WARNING and left out.

    Args:
        words: Each kind, with the word for one of them and for several.
        lead: A verb to open the count line with, such as `renamed`.
        extra: Counts appended to the count line, with no names under them.
    """
    for kind in names.keys() - words.keys():
        logger.warning(
            "No word for kind %r, its %s names are not reported", kind, len(names[kind])
        )
    counts = [
        (len(names.get(kind, ())), singular, plural)
        for kind, (singular, plural) in words.items()
    ]
    counts.extend(extra)
    phrase = count_phrase(counts)
    if not phrase:
        return

    logger.info("%s%s%s", COUNT_INDENT, f"{lead} " if lead else "", phrase)
    for kind, (_, plural) in words.items():
        report_names(logger, plural, names.get(kind, []))
