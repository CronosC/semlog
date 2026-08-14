"""Each subcommand, run the way a user runs it.

Everything here goes through `main(argv)` and checks the exit code, what was
printed and what is on disk afterwards. The functions behind the commands are
also called directly where a command hides a return value worth asserting on.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from semanticlog.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, main
from semanticlog.cli.config import resolve_run_config
from semanticlog.cli.run import (
    build_all,
    build_one,
    choose_case_concepts,
    inspect_input,
    load_and_clean,
    rank_case_concepts,
)
from semanticlog.enrich.declare import DeclareResult
from tests.support.ontology import read_functional, read_rdf

NS = "https://oced"


def read_built(path: Path):
    """Read an ontology the CLI wrote, in whichever syntax the sink used."""
    if path.read_text(encoding="utf-8").startswith("Prefix("):
        return read_functional(path)
    return read_rdf(path, NS)


def record(path: Path) -> dict:
    """Return the provenance record of a written ontology, JSON entries parsed."""
    written = dict(read_built(path).ontology_annotations)
    for key in ("buildConfiguration", "buildInputStatistics", "buildOutputs"):
        if key in written:
            written[key] = json.loads(written[key])
    return written


@pytest.fixture
def cfg(ocel, tmp_path):
    return resolve_run_config(
        {"input": [str(ocel)], "output": str(tmp_path / "onto")},
        command="semlog build fixture",
    )


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def test_a_build_writes_one_ontology_and_exits_zero(ocel, tmp_path):
    out = tmp_path / "onto"

    assert main(["build", str(ocel), "--output", str(out), "-q"]) == EXIT_OK
    assert [p.name for p in out.iterdir()] == ["pipeline.owl"]


def test_the_backmap_is_off_by_default(cfg, ocel):
    build_all(cfg)

    assert not cfg.backmap_path(ocel).exists()


def test_the_backmap_is_written_when_asked_for(ocel, tmp_path):
    cfg = resolve_run_config(
        {
            "input": [str(ocel)],
            "output": str(tmp_path / "onto"),
            "save_backmap": True,
        }
    )
    build_one(cfg, ocel)

    assert cfg.backmap_path(ocel).is_file()
    outputs = record(cfg.ontology_path(ocel))["buildOutputs"]
    assert outputs["backmap"] == str(cfg.backmap_path(ocel))


def test_the_ontology_records_how_it_was_produced(cfg, ocel):
    build_one(cfg, ocel)
    written = record(cfg.ontology_path(ocel))

    assert written["sourceLog"] == str(ocel)
    assert len(written["sourceLogSha256"]) == 64
    assert written["sourceLogBytes"] == str(ocel.stat().st_size)
    assert written["buildCommand"] == "semlog build fixture"
    assert written["buildConfiguration"]["profile"] == "OWL2DL"
    assert written["buildInputStatistics"]["events"] == 12
    assert written["materializeRequested"] == "false"


def test_the_command_line_is_recorded_in_the_ontology(ocel, tmp_path):
    out = tmp_path / "onto"
    main(["build", str(ocel), "--output", str(out), "--case-concepts", "order", "-q"])

    command = record(out / "pipeline.owl")["buildCommand"]
    assert command.startswith("semlog build")
    assert "--case-concepts order" in command


def test_the_ontology_says_where_the_case_concepts_came_from(cfg, ocel, tmp_path):
    build_one(cfg, ocel)
    assert record(cfg.ontology_path(ocel))["caseConceptSource"] == "auto(n=3)"

    chosen = resolve_run_config(
        {
            "input": [str(ocel)],
            "output": str(tmp_path / "explicit"),
            "case_concepts": ["order"],
        }
    )
    build_one(chosen, ocel)
    written = record(chosen.ontology_path(ocel))

    assert written["caseConceptSource"] == "config"
    assert written["buildConfiguration"]["case_concepts"] == ["order"]


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


def test_a_build_takes_its_name_and_settings_from_the_config(
    ocel, tmp_path, write_config
):
    config = write_config("hiring-el.toml", f'input = "{ocel}"\nprofile = "OWL2EL"\n')
    out = tmp_path / "onto"

    assert (
        main(["build", "--config", str(config), "--output", str(out), "-q"]) == EXIT_OK
    )
    written = record(out / "hiring-el.owl")
    assert written["buildConfiguration"]["profile"] == "OWL2EL"
    assert written["buildConfigFile"] == str(config)


def test_a_flag_beats_the_config_file(ocel, tmp_path, write_config):
    config = write_config("run.toml", f'input = "{ocel}"\nprofile = "OWL2EL"\n')
    out = tmp_path / "onto"

    main(
        [
            "build",
            "--config",
            str(config),
            "--output",
            str(out),
            "--profile",
            "OWL2DL",
            "-q",
        ]
    )

    assert record(out / "run.owl")["buildConfiguration"]["profile"] == "OWL2DL"


def test_a_positional_input_beats_the_config_file(ocel, tmp_path, write_config):
    config = write_config("run.toml", 'input = "does/not/exist.json"\n')

    code = main(
        [
            "build",
            str(ocel),
            "--config",
            str(config),
            "--output",
            str(tmp_path / "o"),
            "-q",
        ]
    )

    assert code == EXIT_OK


def test_a_dry_run_prints_the_plan_and_writes_nothing(ocel, tmp_path, capsys):
    out = tmp_path / "onto"

    code = main(["build", str(ocel), "--output", str(out), "--dry-run", "-q"])

    assert code == EXIT_OK
    assert "would write" in capsys.readouterr().out
    assert not out.exists()


def test_a_streaming_build_writes_a_complete_ontology_with_its_record(tmp_path, ocel):
    out = tmp_path / "out"

    assert (
        main(["build", str(ocel), "--streaming", "--output", str(out), "-q"]) == EXIT_OK
    )

    text = (out / "pipeline.owl").read_text()
    assert text.startswith("Prefix(")
    assert text.rstrip().endswith(")")
    assert "sourceLogSha256" in text
    assert [p.name for p in out.iterdir()] == ["pipeline.owl"]


@pytest.mark.parametrize("streaming", [True, False])
def test_a_failing_build_leaves_the_previous_ontology_alone(ocel, tmp_path, streaming):
    """A half-written file where a finished one used to be is worse than a failure."""
    out = tmp_path / "onto"
    good = resolve_run_config({"input": [str(ocel)], "output": str(out)})
    build_all(good)
    before = (out / "pipeline.owl").read_bytes()

    # Fails inside the order emitter, once the sink has already written a
    # TBox and every event, which is when a partial file would appear.
    doomed = dataclasses.replace(good, streaming=streaming, case_concepts=("nope",))
    report = build_all(doomed)

    assert not report.ok
    assert (out / "pipeline.owl").read_bytes() == before
    assert [p.name for p in out.iterdir()] == ["pipeline.owl"]


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


def test_a_configuration_warning_is_repeated_when_the_run_starts(
    ocel, tmp_path, caplog
):
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


@pytest.mark.reasoner
def test_materializing_during_a_build_is_recorded_and_grows_the_file(ocel, tmp_path):
    out = tmp_path / "onto"

    code = main(["build", str(ocel), "--output", str(out), "--materialize", "-q"])

    assert code == EXIT_OK
    assert record(out / "pipeline.owl")["materializeRequested"] == "true"


@pytest.mark.reasoner
def test_a_streamed_build_can_be_materialized(ocel, tmp_path):
    """The reasoner reads the written file, so a streamed build has one to give it."""
    plain = tmp_path / "plain"
    materialized = tmp_path / "materialized"
    main(["build", str(ocel), "--streaming", "--output", str(plain), "-q"])

    code = main(
        [
            "build",
            str(ocel),
            "--streaming",
            "--materialize",
            "--output",
            str(materialized),
            "-q",
        ]
    )

    assert code == EXIT_OK
    grown = (materialized / "pipeline.owl").stat().st_size
    assert grown > (plain / "pipeline.owl").stat().st_size


# --------------------------------------------------------------------------
# build: what stops it
# --------------------------------------------------------------------------


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


def test_a_failed_build_exits_one(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "broken.jsonocel").write_text("not valid ocel")

    assert (
        main(["build", str(folder), "--output", str(tmp_path / "onto"), "-q"])
        == EXIT_FAILED
    )


def test_an_oversized_input_is_skipped_and_reported(ocel, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "big.jsonocel").write_bytes(b"x" * (2 * 1024 * 1024))

    cfg = resolve_run_config(
        {"input": [str(folder)], "output": str(tmp_path / "onto"), "size_limit_mb": 1}
    )
    report = build_all(cfg)

    assert report.ok
    assert [p.name for p, _ in report.skipped] == ["big.jsonocel"]
    assert list((tmp_path / "onto").iterdir()) == []


def test_no_command_prints_help_and_exits_two(capsys):
    assert main([]) == EXIT_USAGE
    assert "COMMAND" in capsys.readouterr().out


def test_an_unknown_profile_exits_two():
    assert main(["build", "log.json", "--profile", "OWL2FULL"]) == EXIT_USAGE


def test_a_missing_input_exits_two_with_a_message(tmp_path, capsys):
    assert main(["build", str(tmp_path / "absent.json"), "-q"]) == EXIT_USAGE
    assert "input not found" in capsys.readouterr().err


def test_a_bad_configuration_exits_two_without_a_traceback(ocel, capsys):
    code = main(
        [
            "build",
            str(ocel),
            "--profile",
            "OWL2EL",
            "--order-representations",
            "df",
            "ef",
            "-q",
        ]
    )

    assert code == EXIT_USAGE
    assert "error:" in capsys.readouterr().err


def test_no_streaming_writes_rdf_xml(tmp_path, ocel):
    """The in-memory sink is still there, and rdflib still reads what it writes."""
    out = tmp_path / "onto"

    code = main(["build", str(ocel), "--no-streaming", "--output", str(out), "-q"])

    assert code == EXIT_OK
    assert (out / "pipeline.owl").read_text(encoding="utf-8").startswith("<?xml")
    assert read_rdf(out / "pipeline.owl", NS).individuals


# --------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------


def test_inspect_prints_a_summary_and_writes_nothing(ocel, tmp_path, capsys):
    assert main(["inspect", str(ocel), "-q"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "object types" in out
    assert "--case-concepts" in out
    assert list(tmp_path.iterdir()) == []


def test_inspect_accepts_several_inputs(ocel, capsys):
    main(["inspect", str(ocel), str(ocel), "-q"])

    assert capsys.readouterr().out.count("object types") == 2


def test_inspect_counts_what_the_log_holds(ocel):
    report = inspect_input(ocel, suggest=2)

    assert dict(report.activities) == {"place_order": 4, "pay": 4, "ship_order": 4}
    assert dict(report.object_types) == {"item": 4, "order": 3}
    assert report.suggested_case_concepts == ["item", "order"]
    assert report.time_range is not None


def test_inspect_orders_its_counts_largest_first(ocel):
    counts = [n for _, n in inspect_input(ocel).object_types]

    assert counts == sorted(counts, reverse=True)


def test_inspect_renders_the_names_a_build_would_accept(ocel):
    text = inspect_input(ocel).render()

    assert "--case-concepts item order" in text
    assert str(ocel) in text


def test_inspect_reports_the_names_cleaning_changed(ocel, tmp_path):
    raw = json.loads(ocel.read_text())
    for obj in raw["objects"]:
        if obj["type"] == "order":
            obj["type"] = "sales order"
    renamed = tmp_path / "renamed.jsonocel"
    renamed.write_text(json.dumps(raw))

    assert inspect_input(renamed).renamed.get("sales_order") == "sales order"


def test_configured_case_concepts_follow_a_rename_and_build(ocel, tmp_path):
    """A qualifier can claim the name of the object type it points at."""
    raw = json.loads(ocel.read_text())
    for event in raw["events"]:
        for relationship in event["relationships"]:
            relationship["qualifier"] = "order"
    collided = tmp_path / "collided.jsonocel"
    collided.write_text(json.dumps(raw))

    cfg = resolve_run_config(
        {
            "input": [str(collided)],
            "output": str(tmp_path / "onto"),
            "case_concepts": ["order"],
        }
    )
    store = load_and_clean(collided, [])
    chosen, source = choose_case_concepts(cfg, store)

    assert store.get_object_types() == ["item", "order_"]
    assert chosen == ["order_"]
    assert source == "config"

    build_one(cfg, collided)

    assert "df_order_" in read_built(cfg.ontology_path(collided)).object_properties


def test_auto_selection_takes_the_most_common_object_types(ocel, tmp_path):
    cfg = resolve_run_config(
        {"input": [str(ocel)], "output": str(tmp_path), "auto_case_concepts": 1}
    )
    store = load_and_clean(ocel, [])

    chosen, source = choose_case_concepts(cfg, store)

    assert chosen == store.get_most_common_objects(n=1)
    assert source == "auto(n=1)"


def _unreached_type(store):
    """Add an object type larger than any the events reach."""
    store.add_object("courier", "cid", pd.DataFrame({"cid": ["c1", "c2", "c3"]}))
    return store


def test_auto_selection_passes_over_a_type_no_event_reaches(tiny_store, ocel, tmp_path):
    """It is the largest type, and it has no events to order."""
    store = _unreached_type(tiny_store)
    cfg = resolve_run_config(
        {"input": [str(ocel)], "output": str(tmp_path), "auto_case_concepts": 2}
    )

    chosen, _ = choose_case_concepts(cfg, store)

    assert store.get_most_common_objects(n=1) == ["courier"]
    assert chosen == ["order"]


def test_the_ranking_inspect_suggests_from_leaves_that_type_out(tiny_store):
    """`semlog inspect` prints this list, so it has to match what a build takes."""
    store = _unreached_type(tiny_store)

    assert rank_case_concepts(store, 3) == ["order"]


# --------------------------------------------------------------------------
# mine-declare
# --------------------------------------------------------------------------


class _FakeEnricher:
    """Stands in for the miner, which needs a solver that is not on PyPI."""

    last_call: dict = {}

    def __init__(self, data):
        self.data = data

    def get_constraints_via_discovery(self, **kwargs):
        _FakeEnricher.last_call = kwargs
        return DeclareResult(
            constraints=pd.DataFrame(
                {
                    "case_concept": ["order", "order"],
                    # Real order ids of the fixture log, since a build refuses
                    # constraints whose traces it cannot find.
                    "trace_id": ["ord_1", "ord_2"],
                    "Response__place_order__pay__": [True, False],
                }
            ),
            case_concepts=["order"],
        )


@pytest.fixture
def fake_enricher(monkeypatch):
    import semanticlog.enrich.declare as declare

    monkeypatch.setattr(declare, "DeclareEnricher", _FakeEnricher)
    return _FakeEnricher


def test_mine_declare_writes_a_csv_a_build_can_read(ocel, tmp_path, fake_enricher):
    out = tmp_path / "constraints.csv"

    code = main(["mine-declare", str(ocel), "--output", str(out), "-q"])

    assert code == EXIT_OK
    assert DeclareResult.read(str(out)).case_concepts == ["order"]


def test_mine_declare_records_which_log_the_constraints_came_from(
    ocel, tmp_path, fake_enricher
):
    out = tmp_path / "constraints.csv"
    main(["mine-declare", str(ocel), "--output", str(out), "-q"])

    result = DeclareResult.read(str(out))

    assert result.source_log == str(ocel)
    assert len(result.source_sha256) == 64


def test_mine_declare_defaults_its_output_to_the_datasets_directory(
    ocel, tmp_path, monkeypatch, fake_enricher
):
    monkeypatch.chdir(tmp_path)

    assert main(["mine-declare", str(ocel), "-q"]) == EXIT_OK
    assert (tmp_path / "datasets" / "declare" / "pipeline.csv").is_file()


def test_mine_declare_passes_the_thresholds_and_templates_through(
    ocel, tmp_path, fake_enricher
):
    main(
        [
            "mine-declare",
            str(ocel),
            "--output",
            str(tmp_path / "c.csv"),
            "--constraints",
            "Response",
            "Precedence",
            "--constraint-support",
            "0.7",
            "--itemset-support",
            "0.4",
            "--case-concepts",
            "order",
            "-q",
        ]
    )

    assert fake_enricher.last_call == {
        "case_concepts": ["order"],
        "constraint_set": ["Response", "Precedence"],
        "constraint_support_threshold": 0.7,
        "itemset_support_threshold": 0.4,
    }


def test_mine_declare_chooses_case_concepts_when_none_are_given(
    ocel, tmp_path, fake_enricher
):
    main(["mine-declare", str(ocel), "--output", str(tmp_path / "c.csv"), "-q"])

    assert fake_enricher.last_call["case_concepts"] == ["item", "order"]


def test_mine_declare_without_the_solver_exits_two_with_a_message(
    ocel, tmp_path, monkeypatch, capsys
):
    import semanticlog.enrich.declare as declare

    def refuse(_data):
        raise ImportError("declare4py is not installed. Install the [declare] extra.")

    monkeypatch.setattr(declare, "DeclareEnricher", refuse)

    code = main(["mine-declare", str(ocel), "--output", str(tmp_path / "c.csv"), "-q"])

    assert code == EXIT_USAGE
    assert "declare4py is not installed" in capsys.readouterr().err


def test_a_build_reads_the_constraints_mine_declare_wrote(
    ocel, tmp_path, fake_enricher
):
    constraints = tmp_path / "constraints.csv"
    main(["mine-declare", str(ocel), "--output", str(constraints), "-q"])

    code = main(
        [
            "build",
            str(ocel),
            "--output",
            str(tmp_path / "onto"),
            "--case-concepts",
            "order",
            "--order-representations",
            "df",
            "declare",
            "--declare-file",
            str(constraints),
            "-q",
        ]
    )

    assert code == EXIT_OK
    assert (
        "Response__place_order__pay__"
        in (tmp_path / "onto" / "pipeline.owl").read_text()
    )


def test_a_build_warns_when_the_constraints_are_from_another_log(
    ocel, tmp_path, fake_enricher, caplog
):
    constraints = tmp_path / "constraints.csv"
    main(["mine-declare", str(ocel), "--output", str(constraints), "-q"])

    meta = Path(str(constraints) + ".meta.json")
    stamped = json.loads(meta.read_text())
    stamped["source_sha256"] = "0" * 64
    meta.write_text(json.dumps(stamped))

    main(
        [
            "build",
            str(ocel),
            "--output",
            str(tmp_path / "onto"),
            "--case-concepts",
            "order",
            "--order-representations",
            "df",
            "declare",
            "--declare-file",
            str(constraints),
            "-q",
        ]
    )

    assert "not the file being built" in caplog.text


# --------------------------------------------------------------------------
# reason
# --------------------------------------------------------------------------


@pytest.mark.reasoner
def test_reason_reports_a_built_ontology_as_consistent(ocel, tmp_path):
    out = tmp_path / "onto"
    main(["build", str(ocel), "--output", str(out), "-q"])

    assert main(["reason", str(out / "pipeline.owl"), "--check", "-q"]) == EXIT_OK


@pytest.mark.reasoner
def test_reason_writes_the_materialized_ontology_where_it_is_told(ocel, tmp_path):
    out = tmp_path / "onto"
    main(["build", str(ocel), "--output", str(out), "-q"])
    before = (out / "pipeline.owl").read_bytes()

    code = main(
        [
            "reason",
            str(out / "pipeline.owl"),
            "--output",
            str(tmp_path / "materialized.owl"),
            "-q",
        ]
    )

    assert code == EXIT_OK
    assert (tmp_path / "materialized.owl").is_file()
    assert (out / "pipeline.owl").read_bytes() == before, "the input is left alone"


@pytest.mark.reasoner
def test_reason_materializes_in_place_without_an_output(ocel, tmp_path):
    out = tmp_path / "onto"
    main(["build", str(ocel), "--output", str(out), "-q"])
    before = (out / "pipeline.owl").stat().st_size

    assert main(["reason", str(out / "pipeline.owl"), "-q"]) == EXIT_OK
    assert (out / "pipeline.owl").stat().st_size > before


def test_reason_on_a_missing_file_exits_two(tmp_path, capsys):
    assert main(["reason", str(tmp_path / "absent.owl"), "--check", "-q"]) == EXIT_USAGE
    assert "not found" in capsys.readouterr().err


@pytest.mark.reasoner
def test_reason_on_an_unreadable_ontology_exits_two_without_a_traceback(
    tmp_path, capsys
):
    """A backend that refuses the file is the caller's problem, not a defect."""
    path = tmp_path / "not-an-ontology.owl"
    path.write_text("this is not an ontology", encoding="utf-8")

    code = main(["reason", str(path), "--check", "-q"])

    assert code == EXIT_USAGE
    assert "will not read" in capsys.readouterr().err


# --------------------------------------------------------------------------
# The report a run hands back
# --------------------------------------------------------------------------


def test_the_report_counts_what_happened(cfg, ocel):
    report = build_all(cfg)

    assert report.ok
    assert report.built == [ocel]
    assert report.summary() == "1 built"
    assert dataclasses.asdict(report)["records"]
