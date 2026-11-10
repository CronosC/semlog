"""The `build`, `mine-declare` and `reason` commands, run through `main` or the functions behind them."""

from __future__ import annotations

import dataclasses
import json
import logging
import re

import pytest

from semanticlog.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, main
from semanticlog.cli.config import resolve_run_config
from semanticlog.cli.run import (
    build_all,
    build_one,
    mine_declare,
)
from semanticlog.core.errors import ConfigError, DependencyError
from semanticlog.core.vocabulary import default_iri
from semanticlog.data.declare import ConceptConstraints, DeclareResult, MiningSettings
from semanticlog.data.loading import load_and_clean
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.pipeline import build_log
from semanticlog.onto.settings import OntoBuildSettings
from tests.support.ocel_files import small_log, write_log
from tests.support.owlapi import load
from tests.support.rdf import read_rdf
from tests.support.written import read_built, record

# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def test_a_build_writes_one_ontology_and_prints_its_path(ocel, tmp_path, capsys):
    out = tmp_path / "onto"

    assert main(["build", str(ocel), "--output", str(out), "-q"]) == EXIT_OK
    assert [p.name for p in out.iterdir()] == ["pipeline.ofn"]
    assert capsys.readouterr().out.strip() == str(out / "pipeline.ofn")


def test_the_header_holds_the_tool_the_settings_and_the_case_concepts(ocel, tmp_path):
    cfg = resolve_run_config(
        {
            "input": [str(ocel)],
            "output": str(tmp_path / "onto"),
            "case_concepts": ["order"],
        }
    )
    build_one(cfg, ocel)
    written = record(cfg.ontology_path(ocel))

    assert set(written) == {"buildTool", "buildConfiguration", "caseConcepts"}
    assert written["buildConfiguration"]["profile"] == "OWL2DL"
    assert written["caseConcepts"] == ["order"]


def test_a_build_from_python_writes_the_header_the_cli_writes(ocel, tmp_path):
    cfg = resolve_run_config(
        {
            "input": [str(ocel)],
            "output": str(tmp_path / "onto"),
            "case_concepts": ["order"],
        }
    )
    build_one(cfg, ocel)
    data = load_and_clean(ocel)
    output = tmp_path / "python.ofn"

    build_log(data, cfg.settings, BuildInputs(("order",)), output)

    assert record(output) == record(cfg.ontology_path(ocel))


def test_a_build_takes_its_name_and_settings_from_the_config(
    ocel, tmp_path, write_config
):
    config = write_config("hiring-el.toml", f'input = "{ocel}"\nprofile = "OWL2EL"\n')
    out = tmp_path / "onto"

    assert (
        main(["build", "--config", str(config), "--output", str(out), "-q"]) == EXIT_OK
    )
    assert record(out / "hiring-el.ofn")["buildConfiguration"]["profile"] == "OWL2EL"


def test_the_backmap_is_written_when_asked_for(ocel, tmp_path):
    cfg = resolve_run_config(
        {"input": [str(ocel)], "output": str(tmp_path / "onto"), "save_backmap": True}
    )
    build_one(cfg, ocel)

    assert cfg.backmap_path(ocel).is_file()


def test_two_builds_of_one_input_are_byte_for_byte_identical(ocel, tmp_path):
    outputs = []
    for name in ("first", "second"):
        cfg = resolve_run_config({"input": [str(ocel)], "output": str(tmp_path / name)})
        build_one(cfg, ocel)
        outputs.append(cfg.ontology_path(ocel).read_bytes())

    assert outputs[0] == outputs[1]


def test_no_streaming_writes_rdf_xml(tmp_path, ocel):
    out = tmp_path / "onto"

    assert (
        main(["build", str(ocel), "--no-streaming", "--output", str(out), "-q"])
        == EXIT_OK
    )
    assert (out / "pipeline.owl").read_text(encoding="utf-8").startswith("<?xml")
    assert "order" in read_rdf(out / "pipeline.owl", default_iri("pipeline")).classes


def test_a_flat_csv_builds_with_its_columns_named_by_flag(flat_csv, tmp_path):
    out = tmp_path / "onto"

    code = main(
        [
            "build",
            str(flat_csv),
            "--case-col",
            "case_id",
            "--act-col",
            "activity",
            "--time-col",
            "timestamp",
            "--object-cols",
            "resource",
            "--case-concepts",
            "case_id",
            "--output",
            str(out),
            "-q",
        ]
    )

    assert code == EXIT_OK
    assert {"case_id", "resource"} <= read_built(out / "flat.ofn").classes


def test_configured_case_concepts_follow_a_rename(ocel, tmp_path):
    """`order` derives `df_order`, which another object type holds, so `order` is renamed and a configured case concept follows."""
    raw = json.loads(ocel.read_text())
    for declared in raw["objectTypes"]:
        if declared["name"] == "item":
            declared["name"] = "df_order"
    for obj in raw["objects"]:
        if obj["type"] == "item":
            obj["type"] = "df_order"
    collided = tmp_path / "collided.jsonocel"
    collided.write_text(json.dumps(raw))
    cfg = resolve_run_config(
        {
            "input": [str(collided)],
            "output": str(tmp_path / "onto"),
            "case_concepts": ["order"],
            "order_representations": ["df"],
        }
    )

    build_one(cfg, collided)

    assert (
        "df_object_order" in read_built(cfg.ontology_path(collided)).object_properties
    )


@pytest.mark.parametrize("streaming", [True, False])
def test_a_failing_build_leaves_the_previous_ontology_alone(ocel, tmp_path, streaming):
    out = tmp_path / "onto"
    good = resolve_run_config(
        {
            "input": [str(ocel)],
            "output": str(out),
            "streaming": streaming,
            "order_representations": ["df"],
        }
    )
    build_all(good)
    path = good.ontology_path(ocel)
    before = path.read_bytes()

    # Fails inside the order emitter, after the TBox and every event are written.
    report = build_all(dataclasses.replace(good, case_concepts=("nope",)))

    assert [p.name for p, _ in report.failed] == ["pipeline.jsonocel"]
    assert path.read_bytes() == before
    assert [p.name for p in out.iterdir()] == [path.name]


def test_a_failure_before_the_build_leaves_no_partial_file(ocel, tmp_path, monkeypatch):
    cfg = resolve_run_config(
        {"input": [str(ocel)], "output": str(tmp_path / "onto"), "streaming": True}
    )

    def fail(*args, **kwargs):
        raise RuntimeError("no build")

    monkeypatch.setattr("semanticlog.cli.run._warn_on_changed_settings", fail)

    with pytest.raises(RuntimeError, match="no build"):
        build_one(cfg, ocel)

    assert list((tmp_path / "onto").glob("*.partial")) == []


@pytest.mark.parametrize("streaming", ["--streaming", "--no-streaming"])
def test_a_rebuild_under_other_settings_warns_before_overwriting(
    ocel, tmp_path, caplog, streaming
):
    common = ["build", str(ocel), "--output", str(tmp_path / "onto"), streaming, "-q"]
    main([*common, "--case-concepts", "order"])
    caplog.clear()

    main([*common, "--case-concepts", "item"])

    assert "different settings (case_concepts)" in caplog.text


def test_a_rebuild_over_a_header_the_owl_api_wrote_warns_before_overwriting(
    ocel, tmp_path, caplog
):
    common = ["build", str(ocel), "--output", str(tmp_path / "onto"), "-q"]
    main([*common, "--case-concepts", "order"])
    path = tmp_path / "onto" / "pipeline.ofn"
    tbox = re.escape(OntoBuildSettings().tbox_iri)
    text = path.read_text(encoding="utf-8")
    # The OWL API writes header properties as prefixed names.
    owlapi = re.sub(rf"Annotation\(<{tbox}(\w+)>", r"Annotation(oced:\1", text)
    assert owlapi != text
    path.write_text(owlapi, encoding="utf-8")
    caplog.clear()

    main([*common, "--case-concepts", "item"])

    assert "different settings (case_concepts)" in caplog.text


def test_rebuilding_the_same_run_does_not_warn_about_overwriting(
    ocel, tmp_path, caplog
):
    argv = [
        "build",
        str(ocel),
        "--output",
        str(tmp_path / "onto"),
        "--case-concepts",
        "order",
        "-q",
    ]
    main(argv)
    caplog.clear()

    main(argv)

    assert "overwritten" not in caplog.text


def test_a_rebuild_over_a_header_without_case_concepts_does_not_warn(
    ocel, tmp_path, caplog
):
    common = ["build", str(ocel), "--output", str(tmp_path / "onto"), "-q"]
    main([*common, "--case-concepts", "order"])
    path = tmp_path / "onto" / "pipeline.ofn"
    text = path.read_text(encoding="utf-8")
    # A header written before the case concepts were recorded in it.
    older = re.sub(r"Annotation\(<[^>]*caseConcepts> \"[^\n]*\"\)\n", "", text)
    assert older != text
    path.write_text(older, encoding="utf-8")
    caplog.clear()

    main([*common, "--case-concepts", "item"])

    assert "overwritten" not in caplog.text


def test_a_run_writes_one_log_file_per_input(ocel, tmp_path):
    cfg = resolve_run_config(
        {
            "input": [str(ocel)],
            "output": str(tmp_path / "onto"),
            "log_dir": str(tmp_path / "logs"),
        }
    )
    build_all(cfg)

    assert len(list((tmp_path / "logs").glob("*_pipeline.log"))) == 1


def test_a_configuration_warning_is_logged_when_the_run_starts(ocel, tmp_path, caplog):
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


def test_a_broken_input_does_not_stop_the_batch(ocel, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "good.jsonocel").write_bytes(ocel.read_bytes())
    (folder / "broken.jsonocel").write_text("not valid ocel")
    cfg = resolve_run_config({"input": [str(folder)], "output": str(tmp_path / "onto")})

    report = build_all(cfg)

    assert [p.name for p in report.built] == ["good.jsonocel"]
    assert [p.name for p, _ in report.failed] == ["broken.jsonocel"]
    assert report.summary() == "1 built, 1 failed"


def test_a_log_build_cannot_read_is_reported_once_naming_its_path_once(
    tmp_path, capsys
):
    broken = tmp_path / "broken.jsonocel"
    broken.write_text("{ not json")

    code = main(["build", str(broken), "--output", str(tmp_path / "onto"), "-q"])

    lines = capsys.readouterr().err.splitlines()
    assert code == EXIT_FAILED
    assert len(lines) == 1
    assert lines[0].count("broken.jsonocel") == 1
    assert "is not well-formed JSON" in lines[0]


def test_a_semlog_error_in_one_input_is_logged_without_a_traceback(
    ocel, tmp_path, monkeypatch, caplog
):
    cfg = resolve_run_config({"input": [str(ocel)], "output": str(tmp_path / "onto")})

    def reject(*args, **kwargs):
        raise ConfigError("Case concept 'nope' is not an object type.")

    monkeypatch.setattr("semanticlog.cli.run.choose_case_concepts", reject)

    report = build_all(cfg)

    assert report.failed == [(ocel, "Case concept 'nope' is not an object type.")]
    assert "Failed while processing pipeline.jsonocel" in caplog.text
    assert "Traceback" not in caplog.text


def test_a_defect_in_one_input_is_logged_with_its_traceback(
    ocel, tmp_path, monkeypatch, caplog
):
    cfg = resolve_run_config({"input": [str(ocel)], "output": str(tmp_path / "onto")})

    def burst(*args, **kwargs):
        raise RuntimeError("a defect")

    monkeypatch.setattr("semanticlog.cli.run.choose_case_concepts", burst)

    report = build_all(cfg)

    assert report.failed == [(ocel, "a defect")]
    assert "Traceback" in caplog.text


def test_an_oversized_input_is_skipped_and_reported(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "big.jsonocel").write_bytes(b"x" * (2 * 1024 * 1024))
    cfg = resolve_run_config(
        {"input": [str(folder)], "output": str(tmp_path / "onto"), "size_limit_mb": 1}
    )

    report = build_all(cfg)

    assert report.ok is True
    assert report.summary() == "0 built, 1 skipped"
    assert list((tmp_path / "onto").iterdir()) == []


@pytest.mark.reasoner
def test_materializing_during_a_build_grows_the_file(ocel, tmp_path):
    main(["build", str(ocel), "--output", str(tmp_path / "plain"), "-q"])

    code = main(
        ["build", str(ocel), "--materialize", "--output", str(tmp_path / "grown"), "-q"]
    )

    assert code == EXIT_OK
    grown = tmp_path / "grown" / "pipeline.ofn"
    assert grown.stat().st_size > (tmp_path / "plain" / "pipeline.ofn").stat().st_size


@pytest.mark.reasoner
def test_a_build_writes_each_output_format_and_prints_its_path(ocel, tmp_path, capsys):
    out = tmp_path / "onto"
    argv = ["build", str(ocel), "--output", str(out), "-q"]

    assert main([*argv, "--output-formats", "ttl", "owx"]) == EXIT_OK
    assert sorted(p.name for p in out.iterdir()) == [
        "pipeline.ofn",
        "pipeline.owx",
        "pipeline.ttl",
    ]
    assert capsys.readouterr().out.split() == [
        str(out / "pipeline.ofn"),
        str(out / "pipeline.ttl"),
        str(out / "pipeline.owx"),
    ]


@pytest.mark.reasoner
def test_the_converted_files_hold_what_the_reasoner_added(ocel, tmp_path):
    out = tmp_path / "onto"
    argv = ["build", str(ocel), "--output", str(out), "--materialize", "-q"]

    assert main([*argv, "--output-formats", "owx"]) == EXIT_OK
    assert load(out / "pipeline.owx").getLogicalAxiomCount() == (
        load(out / "pipeline.ofn").getLogicalAxiomCount()
    )


def test_no_progress_line_of_a_build_ends_in_two_dots(ocel, tmp_path, caplog):
    argv = ["build", str(ocel), "--output", str(tmp_path / "onto")]
    with caplog.at_level(logging.INFO, logger="semanticlog"):
        main([*argv, "--save-backmap", "--output-formats", "ttl"])

    assert len(caplog.records) > 10
    assert [
        r.getMessage() for r in caplog.records if r.getMessage().endswith("..")
    ] == []


# --------------------------------------------------------------------------
# mine-declare
# --------------------------------------------------------------------------


class _FakeEnricher:
    """Fake `DeclareEnricher`."""

    last_call: dict = {}

    def __init__(self, data):
        self.data = data

    def get_constraints_via_discovery(self, **kwargs):
        _FakeEnricher.last_call = kwargs
        # Order ids of the fixture log, since a build rejects constraints whose
        # traces it cannot find.
        return DeclareResult(
            concepts=(
                ConceptConstraints(
                    "order",
                    ("ord_1", "ord_2"),
                    {"Response__place_order__pay__": frozenset({"ord_1"})},
                ),
            ),
            mining=MiningSettings("discovery", None, 0.5, 0.5),
        )


@pytest.fixture
def fake_enricher(monkeypatch):
    import semanticlog.enrich.declare as declare

    monkeypatch.setattr(declare, "DeclareEnricher", _FakeEnricher)
    return _FakeEnricher


def test_mine_declare_writes_constraints_that_record_their_log(
    ocel, tmp_path, fake_enricher
):
    out = tmp_path / "constraints.declare.json"

    assert (
        main(["mine-declare", str(ocel), "--declare-file", str(out), "-q"]) == EXIT_OK
    )
    result = DeclareResult.read(out)
    assert result.case_concepts == ["order"]
    assert result.source_log == str(ocel)
    assert result.mining == MiningSettings("discovery", None, 0.5, 0.5)


def test_mine_declare_logs_the_constraint_count_of_each_case_concept(
    ocel, tmp_path, fake_enricher, caplog
):
    with caplog.at_level(logging.INFO, logger="semanticlog"):
        mine_declare(ocel, tmp_path / "c.declare.json")

    assert "order: 1 DECLARE constraint(s) over 2 trace(s)" in caplog.text
    assert "Saved 1 DECLARE constraint(s) to" in caplog.text


class _EmptyEnricher(_FakeEnricher):
    """Fake `DeclareEnricher` whose mining keeps no constraint."""

    def get_constraints_via_discovery(self, **kwargs):
        result = super().get_constraints_via_discovery(**kwargs)
        (order,) = result.concepts
        empty = ConceptConstraints(order.case_concept, order.trace_ids, {})
        return dataclasses.replace(result, concepts=(empty,))


def test_mine_declare_warns_when_no_constraint_is_kept(
    ocel, tmp_path, monkeypatch, caplog
):
    import semanticlog.enrich.declare as declare

    monkeypatch.setattr(declare, "DeclareEnricher", _EmptyEnricher)

    with caplog.at_level(logging.WARNING, logger="semanticlog"):
        mine_declare(ocel, tmp_path / "c.declare.json")

    assert "Mining kept no DECLARE constraints" in caplog.text
    assert "itemset_support" in caplog.text


def test_mine_declare_passes_the_thresholds_and_templates_through(
    ocel, tmp_path, fake_enricher
):
    main(
        [
            "mine-declare",
            str(ocel),
            "--declare-file",
            str(tmp_path / "c.declare.json"),
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


def test_mine_declare_writes_into_the_current_directory_without_an_output(
    ocel, tmp_path, monkeypatch, fake_enricher
):
    monkeypatch.chdir(tmp_path)

    assert main(["mine-declare", str(ocel), "-q"]) == EXIT_OK
    assert (tmp_path / "pipeline.declare.json").is_file()


def test_mine_declare_writes_to_the_declare_file_flag_over_the_config(
    ocel, tmp_path, write_config, fake_enricher
):
    from_config = tmp_path / "declare" / "config.declare.json"
    config = write_config(
        "mine.toml", f'input = "{ocel}"\ndeclare_file = "{from_config}"\n'
    )

    main(["mine-declare", "--config", str(config), "-q"])
    assert from_config.is_file()
    from_config.unlink()

    main(
        [
            "mine-declare",
            "--config",
            str(config),
            "--declare-file",
            str(tmp_path / "named.declare.json"),
            "-q",
        ]
    )
    assert (tmp_path / "named.declare.json").is_file()
    assert not from_config.exists()


def test_mine_declare_rejects_an_unknown_template_before_reading_the_log(
    ocel, tmp_path, capsys, monkeypatch
):
    def unread(*args, **kwargs):
        raise AssertionError("the log was read before the templates were checked")

    monkeypatch.setattr("semanticlog.cli.run.load_and_clean", unread)

    code = main(
        [
            "mine-declare",
            str(ocel),
            "--constraints",
            "Responce",
            "--declare-file",
            str(tmp_path / "c.declare.json"),
            "-q",
        ]
    )

    assert code == EXIT_USAGE
    assert "Unknown DECLARE template(s): 'Responce'" in capsys.readouterr().err


def test_mine_declare_without_the_solver_exits_two_with_a_message(
    ocel, tmp_path, monkeypatch, capsys
):
    import semanticlog.enrich.declare as declare

    def missing(_data):
        raise DependencyError(
            "declare4py is not installed. Install the [declare] extra."
        )

    monkeypatch.setattr(declare, "DeclareEnricher", missing)

    code = main(
        [
            "mine-declare",
            str(ocel),
            "--declare-file",
            str(tmp_path / "c.declare.json"),
            "-q",
        ]
    )

    assert code == EXIT_USAGE
    assert "declare4py is not installed" in capsys.readouterr().err


@pytest.mark.slow
def test_a_build_reads_the_constraints_mine_declare_wrote(
    ocel, tmp_path, fake_enricher
):
    constraints = tmp_path / "constraints.declare.json"
    main(["mine-declare", str(ocel), "--declare-file", str(constraints), "-q"])

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
        in read_built(tmp_path / "onto" / "pipeline.ofn").classes
    )


# --------------------------------------------------------------------------
# reason
# --------------------------------------------------------------------------


@pytest.fixture
def built_ofn(ocel, tmp_path):
    out = tmp_path / "onto"
    main(["build", str(ocel), "--output", str(out), "-q"])
    return out / "pipeline.ofn"


@pytest.mark.reasoner
def test_reason_reports_a_built_ontology_as_consistent(built_ofn, capsys):
    capsys.readouterr()

    assert main(["reason", str(built_ofn), "--check", "-q"]) == EXIT_OK
    assert capsys.readouterr().out.strip() == f"{built_ofn}: consistent"


@pytest.mark.reasoner
def test_reason_writes_the_materialized_ontology_where_it_is_told(built_ofn, tmp_path):
    before = built_ofn.read_bytes()
    target = tmp_path / "materialized.owl"

    assert main(["reason", str(built_ofn), "--output", str(target), "-q"]) == EXIT_OK
    assert target.is_file()
    assert built_ofn.read_bytes() == before


@pytest.mark.reasoner
def test_reason_materializes_in_place_without_an_output(built_ofn):
    before = built_ofn.stat().st_size

    assert main(["reason", str(built_ofn), "-q"]) == EXIT_OK
    assert built_ofn.stat().st_size > before


@pytest.mark.reasoner
def test_reason_writes_rdf_xml_into_an_owl_output(built_ofn, tmp_path):
    target = tmp_path / "materialized.owl"

    assert main(["reason", str(built_ofn), "--output", str(target), "-q"]) == EXIT_OK
    assert target.read_text(encoding="utf-8").startswith("<?xml")


def test_reason_into_an_output_of_no_format_exits_two_with_one_line(
    built_ofn, tmp_path, capsys
):
    capsys.readouterr()
    target = tmp_path / "materialized.txt"

    code = main(["reason", str(built_ofn), "--output", str(target), "-q"])

    err = capsys.readouterr().err
    assert code == EXIT_USAGE
    assert len(err.strip().splitlines()) == 1
    assert ".ofn, .owl, .owx, .ttl, .omn" in err
    assert not target.exists()


def test_reason_check_with_an_output_exits_two_with_one_line(tmp_path, capsys):
    target = tmp_path / "materialized.ofn"

    code = main(
        ["reason", str(tmp_path / "x.ofn"), "--check", "--output", str(target), "-q"]
    )

    err = capsys.readouterr().err
    assert code == EXIT_USAGE
    assert len(err.strip().splitlines()) == 1
    assert "--check writes nothing" in err
    assert not target.exists()


def test_reason_on_a_missing_file_exits_two(tmp_path, capsys):
    assert main(["reason", str(tmp_path / "absent.owl"), "--check", "-q"]) == EXIT_USAGE
    assert "not found" in capsys.readouterr().err


def test_reason_rejects_an_unknown_backend_before_reading_the_file(tmp_path, capsys):
    code = main(
        ["reason", str(tmp_path / "absent.owl"), "--reasoner-backend", "Konclude", "-q"]
    )

    assert code == EXIT_USAGE
    assert "--reasoner-backend: unknown value 'Konclude'" in capsys.readouterr().err


@pytest.mark.reasoner
def test_reason_on_an_unreadable_ontology_exits_two(tmp_path, capsys):
    path = tmp_path / "not-an-ontology.owl"
    path.write_text("this is not an ontology", encoding="utf-8")

    assert main(["reason", str(path), "--check", "-q"]) == EXIT_USAGE
    assert "cannot read" in capsys.readouterr().err


def _never_mined(_data):
    raise AssertionError("the log was mined despite unusable case concepts")


def test_mine_declare_rejects_an_unknown_case_concept_naming_the_choices(
    ocel, tmp_path, monkeypatch, capsys
):
    import semanticlog.enrich.declare as declare

    monkeypatch.setattr(declare, "DeclareEnricher", _never_mined)

    code = main(
        [
            "mine-declare",
            str(ocel),
            "--case-concepts",
            "nosuch",
            "--declare-file",
            str(tmp_path / "c.declare.json"),
            "-q",
        ]
    )

    err = capsys.readouterr().err
    assert code == EXIT_USAGE
    assert "'nosuch' is not an object type of this log" in err
    assert re.search(r"Set case_concepts to one of: \w+", err) is not None


def test_mine_declare_on_a_log_without_event_to_object_relations_names_none(
    tmp_path, monkeypatch, capsys
):
    import semanticlog.enrich.declare as declare

    monkeypatch.setattr(declare, "DeclareEnricher", _never_mined)
    log = small_log()
    for event in log["events"]:
        event["relationships"] = []

    code = main(
        [
            "mine-declare",
            write_log(tmp_path, log),
            "--declare-file",
            str(tmp_path / "c.declare.json"),
            "-q",
        ]
    )

    assert code == EXIT_USAGE
    assert (
        "No event-to-object relation of this log names an object type, so "
        "case_concepts cannot name one." in capsys.readouterr().err
    )
