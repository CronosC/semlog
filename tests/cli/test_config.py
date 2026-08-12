"""Resolving defaults, TOML files and flags into one checked run."""

from pathlib import Path

import pytest

from semanticlog.cli.config import (
    BUILD_DEFAULTS,
    RENAMED_KEYS,
    ConfigError,
    enum_choices,
    load_config_file,
    parse_enum,
    resolve_run_config,
)
from semanticlog.onto.settings import (
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OCELExtension,
    OWLProfile,
)


def resolve(ocel: Path, **overrides):
    return resolve_run_config({"input": [str(ocel)], **overrides})


@pytest.mark.parametrize(
    ("enum_cls", "token", "expected"),
    [
        (OWLProfile, "OWL2EL", OWLProfile.OWL2EL),
        (OWLProfile, "owl2el", OWLProfile.OWL2EL),
        (EventOrderRepresentation, "DF", EventOrderRepresentation.DF),
        (EventOrderRepresentation, "df", EventOrderRepresentation.DF),
        (
            EventOrderRepresentation,
            "EF_EXPLICIT",
            EventOrderRepresentation.EF_EXPLICIT,
        ),
        (
            EventOrderRepresentation,
            "ef_explicit",
            EventOrderRepresentation.EF_EXPLICIT,
        ),
        (OCELExtension, "ATTRIBUTE_HISTORY", OCELExtension.ATTRIBUTE_HISTORY),
        (OCELExtension, "attribute_history", OCELExtension.ATTRIBUTE_HISTORY),
        (InverseRepresentation, "IMPLICIT", InverseRepresentation.IMPLICIT),
        (DataRepresentation, "equal_bins", DataRepresentation.EQUAL_BINS),
    ],
)
def test_parse_enum_accepts_the_name_and_the_value(enum_cls, token, expected):
    assert parse_enum(enum_cls, token, key="k") is expected


def test_parse_enum_reports_the_key_and_the_alternatives():
    with pytest.raises(
        ConfigError, match="order_representations: unknown value 'nope'"
    ):
        parse_enum(EventOrderRepresentation, "nope", key="order_representations")


def test_enum_choices_lists_values_before_extra_names():
    assert enum_choices(OCELExtension) == ["attribute_history", "ATTRIBUTE_HISTORY"]
    assert enum_choices(OWLProfile) == ["OWL2DL", "OWL2EL"]


def test_load_config_file_reads_a_flat_table(write_config):
    path = write_config(
        "run.toml", 'profile = "OWL2EL"\norder_representations = ["df"]\n'
    )
    assert load_config_file(path) == {
        "profile": "OWL2EL",
        "order_representations": ["df"],
    }


def test_load_config_file_rejects_an_unknown_key_with_a_suggestion(write_config):
    path = write_config("run.toml", 'proflie = "OWL2EL"\n')
    with pytest.raises(ConfigError, match="did you mean 'profile'"):
        load_config_file(path)


@pytest.mark.parametrize(("old", "new"), sorted(RENAMED_KEYS.items()))
def test_a_renamed_key_is_reported_with_its_replacement(write_config, old, new):
    path = write_config("run.toml", f"{old} = []\n")
    with pytest.raises(ConfigError, match=f"did you mean '{new}'"):
        load_config_file(path)


def test_load_config_file_rejects_a_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_config_file(tmp_path / "absent.toml")


def test_load_config_file_reports_invalid_toml(write_config):
    path = write_config("run.toml", "profile = \n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config_file(path)


def test_extends_merges_the_base_and_lets_the_child_win(write_config):
    write_config(
        "_base.toml", 'profile = "OWL2DL"\nbins = 7\nderive_annotations = true\n'
    )
    child = write_config("run.toml", 'extends = "_base.toml"\nprofile = "OWL2EL"\n')

    assert load_config_file(child) == {
        "profile": "OWL2EL",
        "bins": 7,
        "derive_annotations": True,
    }


def test_extends_follows_a_chain(write_config):
    write_config("a.toml", "bins = 1\n")
    write_config("b.toml", 'extends = "a.toml"\nsize_limit_mb = 2\n')
    child = write_config("c.toml", 'extends = "b.toml"\noutput = "out"\n')

    assert load_config_file(child) == {"bins": 1, "size_limit_mb": 2, "output": "out"}


def test_extends_resolves_relative_to_the_file_it_appears_in(tmp_path, write_config):
    nested = tmp_path / "nested"
    nested.mkdir()
    write_config("_base.toml", 'profile = "OWL2EL"\n')
    child = nested / "run.toml"
    child.write_text('extends = "../_base.toml"\n', encoding="utf-8")

    assert load_config_file(child) == {"profile": "OWL2EL"}


def test_extends_cycle_is_reported_rather_than_recursing(write_config):
    write_config("a.toml", 'extends = "b.toml"\n')
    write_config("b.toml", 'extends = "a.toml"\n')
    with pytest.raises(ConfigError, match="cycle"):
        load_config_file(write_config("c.toml", 'extends = "a.toml"\n'))


def test_extends_must_be_a_string(write_config):
    path = write_config("run.toml", "extends = 3\n")
    with pytest.raises(ConfigError, match="must be a path string"):
        load_config_file(path)


def test_a_flag_overrides_the_config_file(ocel, write_config):
    path = write_config("run.toml", f'input = "{ocel}"\nprofile = "OWL2DL"\n')
    cfg = resolve_run_config({"profile": "OWL2EL"}, config_file=path)
    assert cfg.settings.profile is OWLProfile.OWL2EL


def test_the_config_file_overrides_the_built_in_default(ocel, write_config):
    path = write_config("run.toml", f'input = "{ocel}"\nprofile = "OWL2EL"\n')
    cfg = resolve_run_config({}, config_file=path)
    assert cfg.settings.profile is OWLProfile.OWL2EL


def test_untouched_settings_fall_through_to_the_default(ocel):
    cfg = resolve(ocel)
    assert cfg.settings.profile.value == BUILD_DEFAULTS["profile"]
    assert cfg.settings.order_representations == (EventOrderRepresentation.DF,)
    assert cfg.materialize is False
    assert cfg.save_backmap is False


def test_a_directory_contributes_every_ocel_file_in_it(tmp_path, ocel):
    for name in ("a.jsonocel", "b.jsonocel"):
        (tmp_path / name).write_bytes(ocel.read_bytes())
    (tmp_path / "notes.md").write_text("ignored")

    cfg = resolve_run_config({"input": [str(tmp_path)]})
    assert [p.name for p in cfg.inputs] == ["a.jsonocel", "b.jsonocel"]


def test_a_missing_input_is_an_error_not_a_silent_skip(tmp_path):
    with pytest.raises(ConfigError, match="input not found"):
        resolve_run_config({"input": [str(tmp_path / "absent.json")]})


def test_a_directory_without_ocel_files_is_an_error(tmp_path):
    (tmp_path / "notes.md").write_text("nothing to build")
    with pytest.raises(ConfigError, match="no OCEL files"):
        resolve_run_config({"input": [str(tmp_path)]})


def test_no_input_at_all_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="no input given"):
        resolve_run_config({})


def test_the_same_input_named_twice_is_built_once(ocel):
    cfg = resolve_run_config({"input": [str(ocel), str(ocel)]})
    assert cfg.inputs == (ocel,)


def test_a_single_input_takes_its_name_from_the_config_file(ocel, write_config):
    path = write_config("hiring-el-df.toml", f'input = "{ocel}"\n')
    cfg = resolve_run_config({}, config_file=path)
    assert cfg.ontology_path(ocel).name == "hiring-el-df.owl"
    assert cfg.backmap_path(ocel).name == "hiring-el-df.backmap.json"


def test_without_a_config_the_input_name_is_used(ocel):
    cfg = resolve(ocel)
    assert cfg.ontology_path(ocel).name == "pipeline.owl"


def test_a_batch_names_each_output_after_its_input(tmp_path, ocel, write_config):
    for name in ("a.jsonocel", "b.jsonocel"):
        (tmp_path / name).write_bytes(ocel.read_bytes())
    path = write_config("batch.toml", f'input = "{tmp_path}"\n')

    cfg = resolve_run_config({}, config_file=path)
    assert {cfg.ontology_path(p).name for p in cfg.inputs} == {"a.owl", "b.owl"}


def test_an_explicit_name_may_not_collapse_a_batch(tmp_path, ocel):
    for name in ("a.jsonocel", "b.jsonocel"):
        (tmp_path / name).write_bytes(ocel.read_bytes())
    with pytest.raises(ConfigError, match="share one output file"):
        resolve_run_config({"input": [str(tmp_path)], "name": "both"})


def test_axiomatic_inverses_are_rejected_in_owl2el(ocel):
    with pytest.raises(ConfigError, match="AXIOMATIC"):
        resolve(ocel, profile="OWL2EL", inverse_representations=["axiomatic"])


def test_ef_is_rejected_in_owl2el(ocel):
    with pytest.raises(ConfigError, match="EF"):
        resolve(ocel, profile="OWL2EL", order_representations=["df", "ef"])


def test_both_binning_modes_at_once_are_rejected(ocel):
    with pytest.raises(ConfigError, match="mutually exclusive"):
        resolve(ocel, data_representations=["equal_bins", "dist_bins"])


def test_bin_labels_must_match_the_bin_count(ocel):
    with pytest.raises(ConfigError, match="bin_labels"):
        resolve(
            ocel, data_representations=["equal_bins"], bins=3, bin_labels=["a", "b"]
        )


def test_declare_without_a_constraints_file_points_at_mine_declare(ocel):
    with pytest.raises(ConfigError, match="semlog mine-declare"):
        resolve(ocel, order_representations=["declare"])


def test_explicitly_empty_case_concepts_conflict_with_an_order_needing_traces(ocel):
    with pytest.raises(ConfigError, match="case_concepts is empty"):
        resolve(ocel, order_representations=["df"], case_concepts=[])


def test_explicitly_empty_case_concepts_are_fine_without_an_order(ocel):
    cfg = resolve(ocel, order_representations=[], case_concepts=[])
    assert cfg.case_concepts == ()


def test_omitted_case_concepts_mean_auto_selection(ocel):
    assert resolve(ocel).case_concepts is None


def test_bins_without_a_binning_mode_only_warns(ocel):
    cfg = resolve(ocel, bins=3, bin_labels=["a", "b", "c"])
    assert any("no effect" in w for w in cfg.warnings)


def test_a_wrongly_typed_value_names_the_setting(ocel):
    with pytest.raises(ConfigError, match="bins: expected a whole number"):
        resolve(ocel, bins="five")


def test_settings_for_does_not_mutate_the_shared_settings(ocel):
    cfg = resolve(ocel)
    first = cfg.settings_for(["order"])
    second = cfg.settings_for(["item"])

    assert first.case_concepts == ("order",)
    assert second.case_concepts == ("item",)
    assert cfg.settings.case_concepts == ()


def test_render_names_every_output_it_would_write(ocel):
    text = resolve(ocel, output="out", save_backmap=True, save_store=True).render()
    assert "out/pipeline.owl" in text
    assert "out/pipeline.backmap.json" in text
    assert "out/pipeline.oced" in text


def test_render_says_when_case_concepts_are_auto_selected(ocel):
    assert "auto" in resolve(ocel).render()
