"""Resolution of TOML configs and command line flags into a :class:`RunConfig`."""

from pathlib import Path

import pytest

from semanticlog.cli.config import (
    BUILD_DEFAULTS,
    ConfigError,
    describe,
    enum_choices,
    load_config_file,
    parse_enum,
    resolve_run_config,
)
from semanticlog.onto.settings import (
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OcelExtension,
    OwlProfile,
)


def resolve(ocel: Path, **overrides):
    return resolve_run_config({"input": [str(ocel)], **overrides})


# ---------------------------------------------------------------------------
# Enum tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("enum_cls", "token", "expected"),
    [
        (OwlProfile, "OWL2EL", OwlProfile.OWL2EL),
        (OwlProfile, "owl2el", OwlProfile.OWL2EL),
        (EventOrderRepresentation, "DF", EventOrderRepresentation.DF),
        (EventOrderRepresentation, "df", EventOrderRepresentation.DF),
        (
            EventOrderRepresentation,
            "EF_explicit",
            EventOrderRepresentation.EF_explicit,
        ),
        (
            EventOrderRepresentation,
            "ef_explicit",
            EventOrderRepresentation.EF_explicit,
        ),
        (OcelExtension, "ATTRIBUTE_HISTORY", OcelExtension.ATTRIBUTE_HISTORY),
        (OcelExtension, "attribute_history", OcelExtension.ATTRIBUTE_HISTORY),
        (InverseRepresentation, "IMPLICIT", InverseRepresentation.IMPLICIT),
        (DataRepresentation, "equal_bins", DataRepresentation.EQUAL_BINS),
    ],
)
def test_parse_enum_accepts_the_name_and_the_value(enum_cls, token, expected):
    """The enums mix the two conventions, so a user should not have to know which."""
    assert parse_enum(enum_cls, token, key="k") is expected


def test_parse_enum_reports_the_key_and_the_alternatives():
    with pytest.raises(ConfigError, match="order: unknown value 'nope'"):
        parse_enum(EventOrderRepresentation, "nope", key="order")


def test_enum_choices_lists_values_before_extra_names():
    assert enum_choices(OcelExtension) == ["attribute_history", "ATTRIBUTE_HISTORY"]
    assert enum_choices(OwlProfile) == ["OWL2DL", "OWL2EL"]


# ---------------------------------------------------------------------------
# Config files
# ---------------------------------------------------------------------------


def test_load_config_file_reads_a_flat_table(write_config):
    path = write_config("run.toml", 'profile = "OWL2EL"\norder = ["df"]\n')
    assert load_config_file(path) == {"profile": "OWL2EL", "order": ["df"]}


def test_load_config_file_rejects_an_unknown_key_with_a_suggestion(write_config):
    path = write_config("run.toml", 'proflie = "OWL2EL"\n')
    with pytest.raises(ConfigError, match="did you mean 'profile'"):
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
    write_config("b.toml", 'extends = "a.toml"\nsize_limit = 2\n')
    child = write_config("c.toml", 'extends = "b.toml"\noutput = "out"\n')

    assert load_config_file(child) == {"bins": 1, "size_limit": 2, "output": "out"}


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


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------


def test_a_flag_overrides_the_config_file(ocel, write_config):
    path = write_config("run.toml", f'input = "{ocel}"\nprofile = "OWL2DL"\n')
    cfg = resolve_run_config({"profile": "OWL2EL"}, config_file=path)
    assert cfg.settings.profile is OwlProfile.OWL2EL


def test_the_config_file_overrides_the_built_in_default(ocel, write_config):
    path = write_config("run.toml", f'input = "{ocel}"\nprofile = "OWL2EL"\n')
    cfg = resolve_run_config({}, config_file=path)
    assert cfg.settings.profile is OwlProfile.OWL2EL


def test_untouched_settings_fall_through_to_the_default(ocel):
    cfg = resolve(ocel)
    assert cfg.settings.profile.value == BUILD_DEFAULTS["profile"]
    assert cfg.settings.order_representation == [EventOrderRepresentation.DF]
    assert cfg.materialize is False
    assert cfg.save_backmap is False


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------


def test_a_single_input_takes_its_name_from_the_config_file(ocel, write_config):
    path = write_config("hiring-el-df.toml", f'input = "{ocel}"\n')
    cfg = resolve_run_config({}, config_file=path)
    assert cfg.ontology_path(ocel).name == "hiring-el-df.owl"
    assert cfg.manifest_path(ocel).name == "hiring-el-df.manifest.json"


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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_axiomatic_inverses_are_rejected_in_owl2el(ocel):
    with pytest.raises(ConfigError, match="AXIOMATIC"):
        resolve(ocel, profile="OWL2EL", inverse_representation=["axiomatic"])


def test_ef_is_rejected_in_owl2el(ocel):
    with pytest.raises(ConfigError, match="EF"):
        resolve(ocel, profile="OWL2EL", order=["df", "ef"])


def test_both_binning_modes_at_once_are_rejected(ocel):
    with pytest.raises(ConfigError, match="mutually exclusive"):
        resolve(ocel, data_representation=["equal_bins", "dist_bins"])


def test_bin_labels_must_match_the_bin_count(ocel):
    with pytest.raises(ConfigError, match="bin_labels"):
        resolve(ocel, data_representation=["equal_bins"], bins=3, bin_labels=["a", "b"])


def test_declare_without_a_constraints_file_points_at_mine_declare(ocel):
    with pytest.raises(ConfigError, match="semlog mine-declare"):
        resolve(ocel, order=["declare"])


def test_explicitly_empty_case_concepts_conflict_with_an_order_needing_traces(ocel):
    with pytest.raises(ConfigError, match="case_concepts is empty"):
        resolve(ocel, order=["df"], case_concepts=[])


def test_explicitly_empty_case_concepts_are_fine_without_an_order(ocel):
    cfg = resolve(ocel, order=[], case_concepts=[])
    assert cfg.case_concepts == ()


def test_omitted_case_concepts_mean_auto_selection(ocel):
    assert resolve(ocel).case_concepts is None


def test_bins_without_a_binning_mode_only_warns(ocel):
    cfg = resolve(ocel, bins=3, bin_labels=["a", "b", "c"])
    assert any("no effect" in w for w in cfg.warnings)


def test_a_wrongly_typed_value_names_the_setting(ocel):
    with pytest.raises(ConfigError, match="bins: expected a whole number"):
        resolve(ocel, bins="five")


# ---------------------------------------------------------------------------
# settings_for and describe
# ---------------------------------------------------------------------------


def test_settings_for_does_not_mutate_the_shared_settings(ocel):
    cfg = resolve(ocel)
    first = cfg.settings_for(["order"])
    second = cfg.settings_for(["item"])

    assert first.case_concepts == ["order"]
    assert second.case_concepts == ["item"]
    assert cfg.settings.case_concepts == []


def test_describe_names_every_output_it_would_write(ocel):
    text = describe(resolve(ocel, output="out", save_backmap=True, save_store=True))
    assert "out/pipeline.owl" in text
    assert "out/pipeline.manifest.json" in text
    assert "out/pipeline.backmap.json" in text
    assert "out/pipeline.oced" in text


def test_describe_says_when_case_concepts_are_auto_selected(ocel):
    assert "auto" in describe(resolve(ocel))
