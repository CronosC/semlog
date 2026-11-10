"""The `inspect` command, run the way a user runs it."""

from __future__ import annotations

import json

import pytest

from semlog.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, main
from semlog.cli.inspect import inspect_input


def test_inspect_prints_a_summary_and_writes_nothing(ocel, tmp_path, capsys):
    assert main(["inspect", str(ocel), "-q"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "object types" in out
    assert "time range" in out
    assert list(tmp_path.iterdir()) == []


def test_a_log_inspect_cannot_read_does_not_lose_the_others(
    ocel, tmp_path, capsys, caplog
):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "a_good.jsonocel").write_bytes(ocel.read_bytes())
    (folder / "b_broken.jsonocel").write_text("not valid ocel")
    (folder / "c_good.jsonocel").write_bytes(ocel.read_bytes())

    code = main(["inspect", str(folder), "-q"])

    assert code == EXIT_FAILED
    assert capsys.readouterr().out.count("object types") == 2
    assert "b_broken.jsonocel is not well-formed JSON" in caplog.text


def test_a_log_inspect_cannot_read_is_reported_naming_its_path_once(tmp_path, capsys):
    broken = tmp_path / "broken.jsonocel"
    broken.write_text("{ not json")

    code = main(["inspect", str(broken), "-q"])

    lines = capsys.readouterr().err.splitlines()
    assert code == EXIT_FAILED
    assert len(lines) == 1
    assert lines[0].count("broken.jsonocel") == 1
    assert "is not well-formed JSON" in lines[0]


def test_inspect_expands_a_directory_the_way_a_build_does(ocel, tmp_path, capsys):
    for name in ("a.jsonocel", "b.jsonocel"):
        (tmp_path / name).write_bytes(ocel.read_bytes())

    assert main(["inspect", str(tmp_path), "-q"]) == EXIT_OK
    assert capsys.readouterr().out.count("object types") == 2


def test_inspect_reads_the_config_a_build_would(
    flat_csv, tmp_path, capsys, write_config
):
    config = write_config(
        "flat.toml",
        f'input = "{flat_csv}"\n'
        'case_col = "case_id"\n'
        'act_col = "activity"\n'
        'time_col = "timestamp"\n',
    )

    assert main(["inspect", "--config", str(config), "-q"]) == EXIT_OK

    printed = capsys.readouterr().out
    assert "place_order" in printed
    assert "case_id" in printed


def test_inspect_has_no_case_concept_flags(ocel, capsys):
    with pytest.raises(SystemExit) as exit_code:
        main(["inspect", str(ocel), "--case-concepts", "order"])

    assert exit_code.value.code == EXIT_USAGE
    assert "unrecognized arguments" in capsys.readouterr().err


def test_inspect_counts_what_the_log_holds(ocel):
    report = inspect_input(ocel)

    assert dict(report.activities) == {"place_order": 4, "pay": 4, "ship_order": 4}
    assert dict(report.object_types) == {"item": 4, "order": 3}
    assert report.time_range is not None


def test_inspect_orders_its_counts_largest_first(ocel):
    counts = [n for _, n in inspect_input(ocel).object_types]

    assert counts == sorted(counts, reverse=True)


def test_inspect_renders_the_names_a_build_emits(ocel):
    text = inspect_input(ocel).render()

    assert text.startswith(f"{ocel}\n")
    assert "        4  item" in text
    assert "        3  order" in text


def test_inspect_reports_the_names_cleaning_changed(ocel, tmp_path):
    raw = json.loads(ocel.read_text())
    for obj in raw["objects"]:
        if obj["type"] == "order":
            obj["type"] = "sales order"
    renamed = tmp_path / "renamed.jsonocel"
    renamed.write_text(json.dumps(raw))

    assert inspect_input(renamed).renamed.get("sales_order") == "sales order"
