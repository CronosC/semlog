"""Resolving defaults, a TOML file and flags into one checked run."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

from semlog.cli.config import resolve_log_inputs, resolve_run_config
from semlog.core.enums import OCELExtension, OutputFormat, OWLProfile
from semlog.core.errors import ConfigError
from semlog.onto.settings import OntoBuildSettings


def resolve(ocel: Path, **overrides):
    return resolve_run_config({"input": [str(ocel)], **overrides})


def test_a_run_that_names_no_setting_gets_the_library_defaults(ocel):
    assert resolve(ocel).settings == OntoBuildSettings()


def test_a_flag_beats_the_config_file_which_beats_the_default(ocel, write_config):
    path = write_config("run.toml", f'input = "{ocel}"\nprofile = "OWL2DL"\nbins = 7\n')

    cfg = resolve_run_config({"profile": "OWL2EL"}, config_file=path)

    assert (cfg.settings.profile, cfg.settings.bins) == (OWLProfile.OWL2EL, 7)
    assert cfg.settings.derive_annotations is False


def test_an_unknown_value_given_as_a_flag_names_the_flag(ocel):
    with pytest.raises(ConfigError, match="^--profile: unknown value 'foo'"):
        resolve(ocel, profile="foo")


def test_an_unknown_value_in_the_config_names_the_key(ocel, write_config):
    path = write_config("run.toml", f'input = "{ocel}"\nprofile = "foo"\n')

    with pytest.raises(ConfigError, match="^profile: unknown value 'foo'"):
        resolve_run_config({}, config_file=path)


def test_settings_that_contradict_each_other_are_reported_on_one_line(ocel):
    with pytest.raises(ConfigError, match="contradict each other") as failure:
        resolve(ocel, profile="OWL2EL", order_representations=["ef"])

    assert "\n" not in str(failure.value)
    assert "OntoBuildSettings" not in str(failure.value)


def test_a_log_input_setting_given_as_a_flag_names_the_flag(ocel):
    with pytest.raises(ConfigError, match="^--auto-case-concepts: expected"):
        resolve_log_inputs({"input": [str(ocel)], "auto_case_concepts": 0})


def test_settings_are_validated_against_each_other(ocel):
    with pytest.raises(ConfigError, match="ef is not available under OWL2EL"):
        resolve(ocel, profile="OWL2EL", order_representations=["df", "ef"])


def test_a_single_input_takes_its_output_name_from_the_config_file(ocel, write_config):
    cfg = resolve_run_config(
        {}, config_file=write_config("hiring-el-df.toml", f'input = "{ocel}"\n')
    )

    assert cfg.ontology_path(ocel).name == "hiring-el-df.ofn"
    assert cfg.backmap_path(ocel).name == "hiring-el-df.backmap.json"


def test_the_default_iri_holds_the_run_stem(ocel, write_config):
    cfg = resolve_run_config(
        {}, config_file=write_config("hiring el.toml", f'input = "{ocel}"\n')
    )

    assert cfg.log_iri(ocel) == "https://example.org/oced/hiring%20el/"


def test_a_set_iri_is_the_log_base_of_every_input(ocel):
    assert resolve(ocel, iri="https://example.org/x/").log_iri(ocel) == (
        "https://example.org/x/"
    )


def test_a_batch_names_each_output_after_its_input(tmp_path, ocel, write_config):
    for name in ("a.jsonocel", "b.jsonocel"):
        (tmp_path / name).write_bytes(ocel.read_bytes())

    cfg = resolve_run_config(
        {}, config_file=write_config("batch.toml", f'input = "{tmp_path}"\n')
    )

    assert {cfg.ontology_path(p).name for p in cfg.inputs} == {"a.ofn", "b.ofn"}


def test_a_name_for_a_batch_is_rejected(tmp_path, ocel):
    for name in ("a.jsonocel", "b.jsonocel"):
        (tmp_path / name).write_bytes(ocel.read_bytes())

    with pytest.raises(ConfigError, match="would make 2 inputs share one output file"):
        resolve_run_config({"input": [str(tmp_path)], "name": "both"})


def test_a_relative_output_is_kept_as_given(ocel):
    assert resolve(ocel, output="out/nested").output_dir == Path("out/nested")


def test_declare_without_a_constraints_file_points_at_mine_declare(ocel):
    with pytest.raises(
        ConfigError,
        match="includes 'declare' but no declare_file is set.*semlog mine-declare",
    ):
        resolve(ocel, order_representations=["declare"])


def test_a_missing_declare_file_is_rejected_when_it_is_read(ocel, tmp_path):
    cfg = resolve(
        ocel,
        order_representations=["declare"],
        declare_file=str(tmp_path / "nowhere.csv"),
    )

    with pytest.raises(ConfigError, match="declare_file not found"):
        cfg.declare_result()


def test_explicitly_empty_case_concepts_conflict_with_an_order_needing_traces(ocel):
    with pytest.raises(
        ConfigError,
        match="case_concepts is empty but order_representations needs traces",
    ):
        resolve(ocel, order_representations=["df"], case_concepts=[])


def test_empty_case_concepts_are_kept_apart_from_omitted_ones(ocel):
    assert resolve(ocel, order_representations=[], case_concepts=[]).case_concepts == ()
    assert resolve(ocel).case_concepts is None


def test_bins_without_a_binning_mode_only_warn(ocel):
    cfg = resolve(ocel, bins=3, bin_labels=["a", "b", "c"])

    assert (
        len([w for w in cfg.warnings if w.startswith("bins/bin_labels have no effect")])
        == 1
    )


def test_column_names_on_an_ocel_input_only_warn_and_still_reach_the_loader(ocel):
    cfg = resolve(ocel, case_col="case_id", object_cols=["resource"])

    assert (
        len([w for w in cfg.warnings if "have no effect on the OCEL inputs" in w]) == 1
    )
    assert cfg.loader_columns() == {"case_col": "case_id", "object_cols": ["resource"]}


def test_column_names_on_a_flat_input_do_not_warn(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text("case_id,activity,timestamp\nc1,A,2024-01-01 08:00:00\n")

    assert (
        resolve_run_config({"input": [str(path)], "case_col": "case_id"}).warnings == ()
    )


# --------------------------------------------------------------------------
# The annotation document
# --------------------------------------------------------------------------


def test_the_annotation_document_reaches_the_build_inputs(ocel, tmp_path):
    document = tmp_path / "custom.json"
    document.write_text(
        json.dumps({"Event": {"label": "Occurrence"}}), encoding="utf-8"
    )

    inputs = resolve(ocel, annotation_document=str(document)).build_inputs(
        ["order"], None
    )

    assert inputs.annotations == {"Event": {"label": "Occurrence"}}
    assert inputs.case_concepts == ("order",)


def test_a_missing_annotation_document_is_rejected(ocel, tmp_path):
    cfg = resolve(ocel, annotation_document=str(tmp_path / "nowhere.json"))

    with pytest.raises(ConfigError, match="annotation_document not found"):
        cfg.build_inputs(["order"], None)


@pytest.mark.parametrize(
    ("document", "problem"),
    [
        ("{", "invalid JSON"),
        ("[1, 2]", "expected a JSON object"),
        ('{"Event": "Occurrence"}', "entry 'Event': expected an object"),
        ('{"Event": {"title": "x"}}', "unknown key(s) ['title']"),
        ('{"Event": {"label": 3}}', "label must be a string"),
    ],
)
def test_a_malformed_annotation_document_is_rejected_with_its_path(
    ocel, tmp_path, document, problem
):
    path = tmp_path / "custom.json"
    path.write_text(document, encoding="utf-8")
    cfg = resolve(ocel, annotation_document=str(path))

    with pytest.raises(
        ConfigError, match=f"^{re.escape(f'{path}: ')}.*{re.escape(problem)}"
    ):
        cfg.build_inputs(["order"], None)


# --------------------------------------------------------------------------
# The plan a dry run prints
# --------------------------------------------------------------------------


def test_a_dry_run_plan_resolves_to_the_same_run(ocel, tmp_path):
    cfg = resolve(
        ocel,
        output=str(tmp_path),
        profile="OWL2EL",
        case_concepts=["order"],
        name="by-order",
        attribute_representations=["individuals", "equal_bins"],
        bins=2,
        bin_labels=["low", "high"],
        save_backmap=True,
    )
    text = cfg.render()

    again = resolve_run_config(tomllib.loads(text))

    assert again.render() == text
    assert (again.settings, again.inputs, again.name) == (
        cfg.settings,
        cfg.inputs,
        cfg.name,
    )


def test_a_dry_run_plan_names_every_file_it_would_write(ocel):
    text = resolve(ocel, output="out", save_backmap=True).render()

    assert "#   out/pipeline.ofn\n#   out/pipeline.backmap.json" in text


def test_a_dry_run_plan_names_each_converted_file(ocel):
    text = resolve(ocel, output="out", output_formats=["ttl"]).render()

    assert "#   out/pipeline.ofn\n#   out/pipeline.ttl" in text


def test_output_formats_are_read_in_any_case(ocel):
    cfg = resolve(ocel, output_formats=["TTL", "owx"])

    assert cfg.output_formats == [OutputFormat.TTL, OutputFormat.OWX]


def test_an_unknown_output_format_names_the_setting_and_the_choices(ocel):
    with pytest.raises(
        ConfigError,
        match=r"^--output-formats: unknown value 'nt'\. Choose from: ofn, owl, owx, ttl, omn\.",
    ):
        resolve(ocel, output_formats=["nt"])


@pytest.mark.parametrize(
    ("streaming", "requested", "converted"),
    [
        (True, ["ofn", "ttl", "ttl"], ["pipeline.ttl"]),
        (False, ["owl", "ofn"], ["pipeline.ofn"]),
    ],
)
def test_converted_paths_leave_out_the_build_format_and_repeats(
    ocel, streaming, requested, converted
):
    cfg = resolve(ocel, streaming=streaming, output_formats=requested)

    assert [p.name for p in cfg.converted_paths(ocel).values()] == converted


def test_a_dry_run_plan_leaves_out_case_concepts_chosen_from_the_log(ocel):
    assert "\ncase_concepts" not in resolve(ocel).render()
    assert (
        '\ncase_concepts = ["order"]' in resolve(ocel, case_concepts=["order"]).render()
    )


def test_a_dry_run_plan_lists_each_file_of_an_input_directory(tmp_path, ocel):
    for name in ("a.jsonocel", "b.jsonocel"):
        (tmp_path / name).write_bytes(ocel.read_bytes())

    text = resolve_run_config({"input": [str(tmp_path)]}).render()

    assert f'input = ["{tmp_path / "a.jsonocel"}", "{tmp_path / "b.jsonocel"}"]' in text


# --------------------------------------------------------------------------
# Commands that read a log without building
# --------------------------------------------------------------------------


def test_a_command_that_reads_logs_takes_its_settings_from_a_build_config(
    ocel, write_config
):
    path = write_config(
        "run.toml",
        f'input = "{ocel}"\n'
        'case_col = "case_id"\n'
        'case_concepts = ["order"]\n'
        "auto_case_concepts = 1\n"
        'ocel_extensions = ["attribute_history"]\n'
        'declare_file = "out/constraints.csv"\n'
        'constraints = ["Response", "Precedence"]\n'
        "constraint_support = 0.1\n"
        "itemset_support = 0.9\n",
    )

    inputs = resolve_log_inputs({}, config_file=path)

    assert inputs.inputs == (ocel,)
    assert inputs.columns == {"case_col": "case_id"}
    assert inputs.case_concepts == ("order",)
    assert inputs.auto_case_concepts == 1
    assert inputs.ocel_extensions == (OCELExtension.ATTRIBUTE_HISTORY,)
    assert inputs.declare_file == "out/constraints.csv"
    assert inputs.constraints == ["Response", "Precedence"]
    assert (inputs.constraint_support, inputs.itemset_support) == (0.1, 0.9)


def test_a_command_that_reads_one_log_rejects_an_input_of_several(tmp_path, ocel):
    for name in ("a.jsonocel", "b.jsonocel"):
        (tmp_path / name).write_bytes(ocel.read_bytes())

    inputs = resolve_log_inputs({"input": [str(tmp_path)]})

    with pytest.raises(
        ConfigError, match="This command reads one log, but the input resolved to 2"
    ):
        inputs.one_input()
