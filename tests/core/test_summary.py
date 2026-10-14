"""The count line and name lines a build step logs."""

import logging

import pytest

from semanticlog.core.summary import (
    LINE_WIDTH,
    MAX_LINES,
    count_phrase,
    counted,
    name_lines,
    report_by_kind,
    report_names,
)

CLASSES = ("class", "classes")

WORDS = {
    "class": CLASSES,
    "object": ("object property", "object properties"),
}

LOGGER = "semanticlog.test.summary"

HEAD = "    classes:           "

MANY = [f"activity_number_{i}" for i in range(40)]


@pytest.fixture
def logger():
    return logging.getLogger(LOGGER)


def test_a_count_phrase_leaves_out_zero_counts_and_groups_digits():
    phrase = count_phrase(
        [
            (1204553, *CLASSES),
            (0, "object property", "object properties"),
            (1, "axiom", "axioms"),
        ]
    )

    assert phrase == "1,204,553 classes, 1 axiom"


def test_one_count_takes_the_singular_or_plural_word_or_is_empty_at_zero():
    assert [counted(n, *CLASSES) for n in (1, 4, 0)] == ["1 class", "4 classes", ""]


def test_a_short_list_stays_on_one_line_after_its_label():
    assert name_lines("classes", ["Event", "Object", "Time"]) == [
        f"{HEAD}Event, Object, Time"
    ]


def test_a_wrapped_list_hangs_under_its_first_name():
    lines = name_lines("classes", MANY[:6])

    assert len(lines) == 2
    assert lines[0].startswith(f"{HEAD}activity_number_0")
    assert lines[0].endswith(",")
    assert lines[1].startswith(" " * len(HEAD) + "activity_number_")
    assert max(len(line) for line in lines) <= LINE_WIDTH


def test_a_long_list_stops_at_the_line_budget_and_counts_the_names_left_out():
    lines = name_lines("classes", MANY)

    shown = sum(line.count(",") for line in lines[:-1])
    shown += lines[-1].split(", ... ")[0].count(",") + 1
    assert len(lines) == MAX_LINES
    assert lines[-1].endswith(f"(+{len(MANY) - shown} more)")
    assert max(len(line) for line in lines) <= LINE_WIDTH


def test_a_name_longer_than_the_line_is_kept_whole():
    long_name = "a" * (LINE_WIDTH * 2)

    assert name_lines("classes", [long_name]) == [f"{HEAD}{long_name}"]


def test_the_count_of_names_left_out_stays_inside_the_line():
    """A name that fills the line on its own leaves no room for the count."""
    names = ["a" * 56, "b" * 56, "z" * 56, "dropped_one", "dropped_two"]

    lines = name_lines("classes", names)

    assert all(len(line) <= LINE_WIDTH for line in lines)
    assert lines[-1].endswith("(+3 more)")


def test_names_are_logged_one_record_per_line_at_the_given_level(caplog, logger):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        report_names(logger, "classes", MANY[:6], level=logging.WARNING)

    assert [r.levelno for r in caplog.records] == [logging.WARNING, logging.WARNING]


def test_every_name_is_logged_when_debug_is_on(caplog, logger):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        report_names(logger, "classes", MANY)

    assert " ".join(r.message for r in caplog.records).count("activity_number_") == len(
        MANY
    )


def test_a_step_report_is_a_count_line_with_its_lead_and_extras_then_the_names(
    caplog, logger
):
    names = {"class": ["Event", "Object"], "object": ["observed_at"]}

    with caplog.at_level(logging.INFO, logger=LOGGER):
        report_by_kind(
            logger, WORDS, names, lead="renamed", extra=[(91, "axiom", "axioms")]
        )

    messages = [r.message for r in caplog.records]
    assert messages == [
        "  renamed 2 classes, 1 object property, 91 axioms",
        f"{HEAD}Event, Object",
        "    object properties: observed_at",
    ]


def test_a_step_that_produced_nothing_logs_nothing(caplog, logger):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        report_by_kind(logger, WORDS, {})

    assert caplog.records == []


def test_a_kind_with_no_word_is_warned_about_and_left_out(caplog, logger):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        report_by_kind(logger, WORDS, {"class": ["Event"], "unknown": ["x", "y"]})

    assert [(r.levelno, r.message) for r in caplog.records] == [
        (logging.WARNING, "No word for kind 'unknown', its 2 names are not reported"),
        (logging.INFO, "  1 class"),
        (logging.INFO, f"{HEAD}Event"),
    ]
