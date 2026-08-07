"""Tests for the argument handling and path derivation in :mod:`semanticlog.cli`.

Deliberately does not run a build: everything here is pure argument or path logic.
"""

import logging

import pytest

from semanticlog.cli import (
    FileLog,
    _build_settings,
    _output_paths,
    main,
    parse_args,
    select_input_files,
    validate_args,
)
from semanticlog.onto.settings import (
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OcelExtension,
    OwlProfile,
)


def args_for(*argv: str):
    return parse_args(["--folder", "logs", *argv])


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    args = args_for()
    assert args.folder == "logs"
    assert args.files == []
    assert args.profile == "OWL2DL"
    assert args.order == ["DF"]
    assert args.materialize is True
    assert args.data_representation == ["values"]
    assert args.inverse_representation == []
    assert args.size_limit == 0


def test_parse_args_requires_folder():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_rejects_unknown_profile():
    with pytest.raises(SystemExit):
        args_for("--profile", "OWL2FULL")


def test_parse_args_no_materialize():
    assert args_for("--no-materialize").materialize is False


# ---------------------------------------------------------------------------
# validate_args
# ---------------------------------------------------------------------------


def test_validate_args_accepts_defaults():
    validate_args(args_for())


def test_validate_args_rejects_declare_flags_without_declare_order():
    with pytest.raises(SystemExit, match="require DECLARE in --order"):
        validate_args(args_for("--declare-constraints", "Response"))


def test_validate_args_accepts_declare_flags_with_declare_order():
    validate_args(
        args_for("--order", "DF", "DECLARE", "--declare-constraints", "Response")
    )


def test_validate_args_rejects_both_binning_modes():
    with pytest.raises(SystemExit, match="mutually exclusive"):
        validate_args(args_for("--data-representation", "equal_bins", "dist_bins"))


def test_validate_args_rejects_axiomatic_inverses_in_owl2el():
    with pytest.raises(SystemExit, match="AXIOMATIC"):
        validate_args(
            args_for("--profile", "OWL2EL", "--inverse-representation", "AXIOMATIC")
        )


def test_validate_args_rejects_ef_in_owl2el():
    with pytest.raises(SystemExit, match="EF"):
        validate_args(args_for("--profile", "OWL2EL", "--order", "EF"))


def test_validate_args_rejects_mismatched_bin_labels():
    with pytest.raises(SystemExit, match="bin_labels"):
        validate_args(
            args_for(
                "--data-representation",
                "equal_bins",
                "--bins",
                "3",
                "--bin-labels",
                "low",
                "high",
            )
        )


def test_validate_args_warns_when_bins_have_no_effect(caplog):
    with caplog.at_level(logging.WARNING):
        validate_args(args_for("--bins", "7", "--bin-labels", *"abcdefg"))
    assert "no effect" in caplog.text


def test_validate_args_does_not_require_case_concepts():
    """case_concepts are picked per file, so an empty list must not fail early."""
    validate_args(args_for("--order", "DF"))


# ---------------------------------------------------------------------------
# _build_settings
# ---------------------------------------------------------------------------


def test_build_settings_maps_every_flag():
    args = args_for(
        "--profile",
        "OWL2EL",
        "--order",
        "DF",
        "EF_explicit",
        "--data-representation",
        "values",
        "equal_bins",
        "--inverse-representation",
        "IMPLICIT",
        "--derive-annotations",
    )
    settings = _build_settings(args, ["order"], [OcelExtension.ATTRIBUTE_HISTORY])

    assert settings.profile is OwlProfile.OWL2EL
    assert settings.order_representation == [
        EventOrderRepresentation.DF,
        EventOrderRepresentation.EF_explicit,
    ]
    assert settings.data_representation == [
        DataRepresentation.VALUES,
        DataRepresentation.EQUAL_BINS,
    ]
    assert settings.inverse_representation == [InverseRepresentation.IMPLICIT]
    assert settings.case_concepts == ["order"]
    assert settings.instantiate_attribute_history
    assert settings.derive_annotations


def test_build_settings_defaults_to_no_inverses():
    """The CLI and the library agree: no inverse relations unless asked for."""
    settings = _build_settings(args_for(), [], [])
    assert settings.inverse_representation == []
    assert settings.has_inverses is False


# ---------------------------------------------------------------------------
# _output_paths
# ---------------------------------------------------------------------------


def test_output_paths_encode_profile_order_and_representation():
    paths = _output_paths("Hiring", args_for())
    assert paths.onto == "data/onto/Hiring_MAT_OWL2DL_DF_VALUES.owl"
    assert paths.id_map == "data/id_map/Hiring_MAT_OWL2DL_DF_VALUES.json"
    assert paths.oced == "data/oced/Hiring_MAT_OWL2DL_DF_VALUES.pkl"
    assert paths.declare == "data/onto/Hiring_MAT_OWL2DL_DF_VALUES_declare.csv"


def test_output_paths_omit_mat_when_not_materializing():
    paths = _output_paths("Hiring", args_for("--no-materialize"))
    assert "_MAT" not in paths.onto


def test_output_paths_mark_no_order():
    paths = _output_paths("Hiring", args_for("--order"))
    assert "NoOrder" in paths.onto


def test_output_paths_mark_annotated():
    paths = _output_paths("Hiring", args_for("--derive-annotations"))
    assert paths.onto.endswith("_ANNOTATED.owl")


def test_output_paths_include_ocel_extensions():
    paths = _output_paths("Hiring", args_for("--ocel-extensions", "ATTRIBUTE_HISTORY"))
    assert "ATTRIBUTE_HISTORY" in paths.onto


def test_output_paths_mark_absent_data_representation():
    paths = _output_paths("Hiring", args_for("--data-representation"))
    assert "_NONE" in paths.onto


def test_output_paths_respect_output_dirs(tmp_path):
    paths = _output_paths("Hiring", args_for("--output", str(tmp_path)))
    assert paths.onto.startswith(str(tmp_path))


# ---------------------------------------------------------------------------
# select_input_files
# ---------------------------------------------------------------------------


def test_select_input_files_returns_all_when_files_omitted(tmp_path):
    (tmp_path / "a.xml").write_text("")
    (tmp_path / "b.json").write_text("")
    (tmp_path / "sub").mkdir()
    args = parse_args(["--folder", str(tmp_path)])
    assert sorted(select_input_files(args)) == ["a.xml", "b.json"]


def test_select_input_files_filters_to_requested(tmp_path):
    (tmp_path / "a.xml").write_text("")
    (tmp_path / "b.json").write_text("")
    args = parse_args(["--folder", str(tmp_path), "--files", "b.json"])
    assert select_input_files(args) == ["b.json"]


def test_select_input_files_ignores_unknown_names(tmp_path):
    (tmp_path / "a.xml").write_text("")
    args = parse_args(["--folder", str(tmp_path), "--files", "missing.xml"])
    assert select_input_files(args) == []


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_exits_nonzero_when_a_file_fails(tmp_path):
    """A batch where every input fails must not report success."""
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "broken.xml").write_text("not valid ocel")
    argv = [
        "--folder",
        str(folder),
        "--output",
        str(tmp_path / "onto"),
        "--id-map-output",
        str(tmp_path / "ids"),
        "--oced-output",
        str(tmp_path / "oced"),
        "--no-materialize",
    ]
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert "broken.xml" in str(exc.value)


def test_main_skips_files_over_size_limit(tmp_path):
    """An oversized input is skipped, so the run neither fails nor writes output."""
    folder = tmp_path / "in"
    folder.mkdir()
    # Not valid OCEL: reaching the loader at all would raise and fail the run.
    (folder / "big.xml").write_bytes(b"x" * (2 * 1024 * 1024))
    out = tmp_path / "onto"
    argv = [
        "--folder",
        str(folder),
        "--output",
        str(out),
        "--id-map-output",
        str(tmp_path / "ids"),
        "--oced-output",
        str(tmp_path / "oced"),
        "--size-limit",
        "1",
        "--no-materialize",
    ]
    main(argv)
    assert list(out.iterdir()) == []


# ---------------------------------------------------------------------------
# FileLog
# ---------------------------------------------------------------------------


def test_file_log_disabled_returns_none():
    log = FileLog("")
    assert log.switch_to("stem") is None


def test_file_log_writes_and_prunes(tmp_path):
    log = FileLog(str(tmp_path), keep=2)
    try:
        paths = []
        for i in range(4):
            path = log.switch_to("stem")
            assert path is not None
            # Timestamps have second resolution; make each name distinct.
            (tmp_path / f"2000010{i}_000000_stem.log").write_text("")
            paths.append(path)
        log.close()
        remaining = sorted(p.name for p in tmp_path.glob("*_stem.log"))
        assert len(remaining) <= 3
    finally:
        log.close()


def test_file_log_close_is_idempotent(tmp_path):
    log = FileLog(str(tmp_path))
    log.switch_to("stem")
    log.close()
    log.close()
