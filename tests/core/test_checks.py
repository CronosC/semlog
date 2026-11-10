"""Collecting what the pre-build checks found into one message."""

import pytest

from semlog.core.checks import CheckReport, FixIn, raise_reports
from semlog.core.errors import ConfigError, InputError

CONFIG = CheckReport(
    header="The case concepts are wrong:",
    problems=("'a' is not an object type", "'b' is not an object type"),
    advice="Name types the log holds.",
    fix_in=FixIn.CONFIG,
)

INPUT = CheckReport(
    header="The store holds rows that cannot be emitted:",
    problems=("e2o 'rel': 1 event id is in no event table",),
    advice="Fix the source data.",
    fix_in=FixIn.INPUT,
)


def test_checks_that_found_nothing_raise_nothing():
    raise_reports([None, None])


def test_one_report_is_one_line_of_its_header_then_problems_then_advice():
    with pytest.raises(ConfigError, match="^The case concepts are wrong:") as excinfo:
        raise_reports([None, CONFIG, None])

    assert str(excinfo.value) == (
        "The case concepts are wrong: 'a' is not an object type. "
        "'b' is not an object type. Name types the log holds."
    )


def test_a_report_names_its_first_five_problems_and_counts_the_rest():
    report = CheckReport(
        header="Rows are wrong:", problems=tuple(f"row {i}" for i in range(1, 9))
    )

    assert report.render() == (
        "Rows are wrong: row 1. row 2. row 3. row 4. row 5. 3 more not listed."
    )


def test_a_report_with_no_problems_and_no_advice_is_its_header():
    with pytest.raises(
        InputError, match=r"^The constraints have no 'trace_id' column\.$"
    ):
        raise_reports(
            [CheckReport(header="The constraints have no 'trace_id' column.")]
        )


def test_one_input_finding_among_several_reports_makes_an_input_error_in_run_order():
    with pytest.raises(InputError, match="^The case concepts are wrong:") as excinfo:
        raise_reports([CONFIG, INPUT])

    assert str(excinfo.value) == f"{CONFIG.render()} {INPUT.render()}"
