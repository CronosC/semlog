"""Layout of the count and name lines a build step logs."""

from semanticlog.core.summary import LINE_WIDTH, MAX_LINES, count_phrase, name_lines

CLASSES = ("class", "classes")


def test_count_phrase_leaves_out_the_counts_that_are_zero():
    phrase = count_phrase(
        [
            (7, *CLASSES),
            (0, "object property", "object properties"),
            (1, "axiom", "axioms"),
        ]
    )

    assert phrase == "7 classes, 1 axiom"


def test_count_phrase_is_empty_when_nothing_happened():
    assert count_phrase([(0, *CLASSES)]) == ""


def test_count_phrase_groups_the_digits_of_a_large_count():
    assert count_phrase([(1204553, "axiom", "axioms")]) == "1,204,553 axioms"


def test_name_lines_says_nothing_about_an_empty_list():
    assert name_lines("classes", []) == []


def test_name_lines_keeps_a_short_list_on_one_line():
    lines = name_lines("classes", ["Event", "Object", "Time"])

    assert lines == ["    classes:           Event, Object, Time"]


def test_name_lines_hangs_a_wrapped_list_under_its_first_name():
    lines = name_lines("classes", [f"activity_number_{i}" for i in range(6)])

    assert len(lines) == 2
    assert lines[0].startswith("    classes:           activity_number_0")
    assert lines[0].endswith(",")
    assert lines[1].startswith(" " * len("    classes:           "))
    assert max(len(line) for line in lines) <= LINE_WIDTH


def test_name_lines_stops_after_the_line_budget_and_counts_the_rest():
    lines = name_lines("classes", [f"activity_number_{i}" for i in range(40)])

    assert len(lines) == MAX_LINES
    assert lines[-1].endswith("more)")
    assert max(len(line) for line in lines) <= LINE_WIDTH


def test_name_lines_counts_the_names_it_drops_to_fit_the_marker():
    names = [f"activity_number_{i}" for i in range(40)]

    lines = name_lines("classes", names)

    shown = sum(line.count(",") for line in lines[:-1])
    shown += lines[-1].split(", ... ")[0].count(",") + 1
    assert f"(+{len(names) - shown} more)" in lines[-1]


def test_name_lines_prints_every_name_when_asked_for_all_of_them():
    names = [f"activity_number_{i}" for i in range(40)]

    lines = name_lines("classes", names, complete=True)

    assert len(lines) > MAX_LINES
    assert all("more)" not in line for line in lines)
    assert all(name in " ".join(lines) for name in names)


def test_name_lines_keeps_a_name_longer_than_the_line_rather_than_dropping_it():
    long_name = "a" * (LINE_WIDTH * 2)

    lines = name_lines("classes", [long_name])

    assert lines == [f"    classes:           {long_name}"]
