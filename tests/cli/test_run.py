"""What each subcommand does, and what it leaves on disk."""

import json
from pathlib import Path

import pytest

from semanticlog.cli import EXIT_USAGE, main
from semanticlog.cli.config import resolve_run_config
from semanticlog.cli.run import (
    build_all,
    build_one,
    choose_case_concepts,
    inspect_input,
    load_and_clean,
)
from tests.cli.conftest import run_record


@pytest.fixture
def cfg(ocel, tmp_path):
    return resolve_run_config(
        {"input": [str(ocel)], "output": str(tmp_path / "onto")},
        command="semlog build fixture",
    )


def test_a_build_writes_one_file(cfg, ocel):
    report = build_all(cfg)

    assert report.ok
    assert report.built == [ocel]
    assert [p.name for p in cfg.output_dir.iterdir()] == ["pipeline.owl"]


def test_the_optional_artifacts_are_off_by_default(cfg, ocel):
    build_all(cfg)
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
    build_one(cfg, ocel)

    assert cfg.backmap_path(ocel).is_file()
    assert cfg.store_path(ocel).is_dir()
    outputs = run_record(cfg.ontology_path(ocel))["buildOutputs"]
    assert outputs["backmap"] == str(cfg.backmap_path(ocel))
    assert outputs["store"] == str(cfg.store_path(ocel))


def test_the_ontology_records_how_it_was_produced(cfg, ocel):
    build_one(cfg, ocel)
    record = run_record(cfg.ontology_path(ocel))

    assert record["sourceLog"] == str(ocel)
    assert len(record["sourceLogSha256"]) == 64
    assert record["sourceLogBytes"] == str(ocel.stat().st_size)
    assert record["buildCommand"] == "semlog build fixture"
    assert record["buildConfiguration"]["profile"] == "OWL2DL"
    assert record["buildInputStatistics"]["events"] == 12
    assert record["materialized"] == "false"


def test_the_ontology_says_where_the_case_concepts_came_from(cfg, ocel, tmp_path):
    build_one(cfg, ocel)
    auto = run_record(cfg.ontology_path(ocel))
    assert auto["caseConceptSource"] == "auto(n=3)"
    assert auto["buildConfiguration"]["case_concepts"]

    chosen = resolve_run_config(
        {
            "input": [str(ocel)],
            "output": str(tmp_path / "explicit"),
            "case_concepts": ["order"],
        }
    )
    build_one(chosen, ocel)
    record = run_record(chosen.ontology_path(ocel))
    assert record["caseConceptSource"] == "config"
    assert record["buildConfiguration"]["case_concepts"] == ["order"]


def test_two_builds_of_one_input_are_byte_for_byte_identical(ocel, tmp_path):
    outputs = []
    for name in ("first", "second"):
        cfg = resolve_run_config(
            {"input": [str(ocel)], "output": str(tmp_path / name)},
            command="semlog build fixture",
        )
        build_one(cfg, ocel)
        outputs.append(cfg.ontology_path(ocel).read_bytes())

    assert outputs[0] == outputs[1]


def test_two_configs_of_one_input_do_not_overwrite_each_other(ocel, tmp_path):
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
    report = build_all(cfg)

    assert not report.ok
    assert [p.name for p in report.built] == ["good.jsonocel"]
    assert [p.name for p, _ in report.failed] == ["broken.jsonocel"]


def test_an_oversized_input_is_skipped_and_reported(ocel, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "big.jsonocel").write_bytes(b"x" * (2 * 1024 * 1024))

    cfg = resolve_run_config(
        {
            "input": [str(folder)],
            "output": str(tmp_path / "onto"),
            "size_limit_mb": 1,
        }
    )
    report = build_all(cfg)

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
    build_all(cfg)
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
    build_all(cfg)
    assert "no effect" in caplog.text


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
    raw = json.loads(ocel.read_text())
    for obj in raw["objects"]:
        if obj["type"] == "order":
            obj["type"] = "sales order"
    renamed = tmp_path / "renamed.jsonocel"
    renamed.write_text(json.dumps(raw))

    report = inspect_input(renamed)
    assert report.renamed.get("sales_order") == "sales order"


def test_reason_reports_a_built_ontology_as_consistent(cfg, ocel):
    from semanticlog.cli.run import reason

    build_one(cfg, ocel)
    assert reason(
        Path(cfg.ontology_path(ocel)),
        profile=cfg.settings.profile,
        check_only=True,
    )


def test_a_streaming_build_writes_a_complete_ontology_with_its_record(tmp_path, ocel):
    out = tmp_path / "out"
    assert main(["build", str(ocel), "--streaming", "--output", str(out)]) == 0

    ontology = out / "pipeline.owl"
    assert ontology.is_file()
    text = ontology.read_text()
    assert text.startswith("Prefix(")
    assert text.rstrip().endswith(")")

    record = run_record(ontology)
    assert len(record["sourceLogSha256"]) == 64
    assert record["buildInputStatistics"]["events"] == 12


def test_streaming_and_materialize_together_are_refused(tmp_path, ocel):
    code = main(
        ["build", str(ocel), "--streaming", "--materialize", "--output", str(tmp_path)]
    )
    assert code == EXIT_USAGE
