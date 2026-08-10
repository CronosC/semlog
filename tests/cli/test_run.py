"""The pipelines behind the subcommands, exercised against the committed fixture.

Unlike the old CLI tests these do run real builds: the fixture log is small
enough that an end-to-end pass costs a fraction of a second, and a CLI meant to
be the reproducible entry point should be tested by actually reproducing.
"""

import json
from pathlib import Path

import pytest

from semanticlog.cli.config import resolve_run_config
from semanticlog.cli.run import (
    build_one,
    choose_case_concepts,
    inspect_input,
    load_and_clean,
    run_build,
)


@pytest.fixture
def cfg(ocel, tmp_path):
    return resolve_run_config(
        {"input": [str(ocel)], "output": str(tmp_path / "onto")},
        command="semlog build fixture",
    )


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def test_a_build_writes_an_ontology_and_a_manifest(cfg, ocel):
    report = run_build(cfg)

    assert report.ok
    assert report.built == [ocel]
    assert cfg.ontology_path(ocel).is_file()
    assert cfg.manifest_path(ocel).is_file()


def test_the_optional_artifacts_are_off_by_default(cfg, ocel):
    run_build(cfg)
    assert not cfg.backmap_path(ocel).exists()
    assert not cfg.store_path(ocel).exists()


def test_the_optional_artifacts_are_written_when_asked_for(ocel, tmp_path):
    cfg = resolve_run_config(
        {
            "input": [str(ocel)],
            "output": str(tmp_path / "onto"),
            "save_backmap": True,
            "save_store": True,
        }
    )
    record = build_one(cfg, ocel)

    assert cfg.backmap_path(ocel).is_file()
    # The store is a directory of Parquet tables, not a single pickle.
    assert cfg.store_path(ocel).is_dir()
    assert record["outputs"]["backmap"] == str(cfg.backmap_path(ocel))
    assert record["outputs"]["store"] == str(cfg.store_path(ocel))


def test_the_manifest_records_how_the_result_was_produced(cfg, ocel):
    build_one(cfg, ocel)
    record = json.loads(cfg.manifest_path(ocel).read_text())

    assert record["input"]["path"] == str(ocel)
    assert len(record["input"]["sha256"]) == 64
    assert record["command"] == "semlog build fixture"
    assert record["settings"]["profile"] == "OWL2DL"
    assert record["stats"]["events"] == 12
    assert record["stats"]["individuals"] > 0
    assert "build" in record["timings_s"]
    assert record["materialized"] is False


def test_the_manifest_says_where_the_case_concepts_came_from(cfg, ocel, tmp_path):
    build_one(cfg, ocel)
    auto = json.loads(cfg.manifest_path(ocel).read_text())
    assert auto["case_concepts"]["source"] == "auto(n=3)"
    assert auto["case_concepts"]["value"]

    chosen = resolve_run_config(
        {
            "input": [str(ocel)],
            "output": str(tmp_path / "explicit"),
            "case_concepts": ["order"],
        }
    )
    record = build_one(chosen, ocel)
    assert record["case_concepts"] == {"value": ["order"], "source": "config"}


def test_two_configs_of_one_input_do_not_overwrite_each_other(ocel, tmp_path):
    """The old flag-encoded filenames collided whenever case concepts differed."""
    common = {"input": [str(ocel)], "output": str(tmp_path / "onto")}
    first = resolve_run_config(
        {**common, "case_concepts": ["order"], "name": "by-order"}
    )
    second = resolve_run_config(
        {**common, "case_concepts": ["item"], "name": "by-item"}
    )

    build_one(first, ocel)
    build_one(second, ocel)

    assert (tmp_path / "onto" / "by-order.owl").is_file()
    assert (tmp_path / "onto" / "by-item.owl").is_file()


def test_a_failing_input_does_not_stop_the_batch(ocel, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "good.jsonocel").write_bytes(ocel.read_bytes())
    (folder / "broken.jsonocel").write_text("not valid ocel")

    cfg = resolve_run_config({"input": [str(folder)], "output": str(tmp_path / "onto")})
    report = run_build(cfg)

    assert not report.ok
    assert [p.name for p in report.built] == ["good.jsonocel"]
    assert [p.name for p, _ in report.failed] == ["broken.jsonocel"]


def test_an_oversized_input_is_skipped_and_reported(ocel, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    # Not valid OCEL: reaching the loader at all would fail rather than skip.
    (folder / "big.jsonocel").write_bytes(b"x" * (2 * 1024 * 1024))

    cfg = resolve_run_config(
        {
            "input": [str(folder)],
            "output": str(tmp_path / "onto"),
            "size_limit": 1,
        }
    )
    report = run_build(cfg)

    assert report.ok
    assert [p.name for p, _ in report.skipped] == ["big.jsonocel"]
    assert list((tmp_path / "onto").iterdir()) == []


def test_a_run_writes_one_log_file_per_input(ocel, tmp_path):
    cfg = resolve_run_config(
        {
            "input": [str(ocel)],
            "output": str(tmp_path / "onto"),
            "log_dir": str(tmp_path / "logs"),
        }
    )
    run_build(cfg)
    assert list((tmp_path / "logs").glob("*_pipeline.log"))


def test_the_binning_warning_is_emitted_at_run_time(ocel, tmp_path, caplog):
    cfg = resolve_run_config(
        {
            "input": [str(ocel)],
            "output": str(tmp_path / "onto"),
            "bins": 3,
            "bin_labels": ["a", "b", "c"],
        }
    )
    run_build(cfg)
    assert "no effect" in caplog.text


# ---------------------------------------------------------------------------
# case concepts
# ---------------------------------------------------------------------------


def test_auto_selection_takes_the_most_common_object_types(ocel, tmp_path):
    cfg = resolve_run_config(
        {
            "input": [str(ocel)],
            "output": str(tmp_path),
            "auto_case_concepts": 1,
        }
    )
    store = load_and_clean(ocel, [])
    chosen, source = choose_case_concepts(cfg, store)

    assert chosen == store.get_most_common_objects(n=1)
    assert source == "auto(n=1)"


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def test_inspect_summarises_without_writing_anything(ocel, tmp_path):
    report = inspect_input(ocel, suggest=2)

    assert dict(report.activities) == {"place_order": 4, "pay": 4, "ship_order": 4}
    assert dict(report.object_types) == {"item": 4, "order": 3}
    assert report.suggested_case_concepts == ["item", "order"]
    assert report.time_range is not None
    assert list(tmp_path.iterdir()) == []


def test_inspect_counts_are_ordered_largest_first(ocel):
    counts = [n for _, n in inspect_input(ocel).object_types]
    assert counts == sorted(counts, reverse=True)


def test_inspect_renders_the_names_a_build_would_accept(ocel):
    text = inspect_input(ocel).render()
    assert "--case-concepts item order" in text
    assert str(ocel) in text


def test_inspect_reports_names_that_cleaning_changed(tmp_path, ocel):
    """The cleaned name is what --case-concepts matches, so it is shown first."""
    raw = json.loads(ocel.read_text())
    for obj in raw["objects"]:
        if obj["type"] == "order":
            obj["type"] = "sales order"
    renamed = tmp_path / "renamed.jsonocel"
    renamed.write_text(json.dumps(raw))

    report = inspect_input(renamed)
    assert report.renamed.get("sales_order") == "sales order"


# ---------------------------------------------------------------------------
# reason
# ---------------------------------------------------------------------------


def test_reason_reports_a_built_ontology_as_consistent(cfg, ocel):
    from semanticlog.cli.run import reason

    build_one(cfg, ocel)
    assert reason(
        Path(cfg.ontology_path(ocel)),
        profile=cfg.settings.profile,
        check_only=True,
    )
