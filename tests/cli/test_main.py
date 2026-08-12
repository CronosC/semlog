"""Argument parsing, subcommand dispatch and exit codes."""

import pytest

from semanticlog.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, build_parser, main
from tests.cli.conftest import run_record


def parse(*argv: str):
    return build_parser().parse_args(list(argv))


def test_no_command_prints_help_and_exits_nonzero(capsys):
    assert main([]) == EXIT_USAGE
    assert "COMMAND" in capsys.readouterr().out


def test_every_subcommand_is_reachable():
    for command in ("build", "inspect", "mine-declare", "reason"):
        assert command in build_parser().format_help()


def test_build_flags_default_to_none_so_a_config_can_supply_them():
    args = parse("build", "log.json")
    assert args.profile is None
    assert args.order_representations is None
    assert args.materialize is None
    assert args.case_concepts is None
    assert args.save_backmap is None


def test_an_unknown_profile_is_rejected_with_exit_two():
    assert main(["build", "log.json", "--profile", "OWL2FULL"]) == EXIT_USAGE


def test_a_profile_is_accepted_in_either_spelling_and_any_case():
    for spelling in ("OWL2DL", "owl2dl", "Owl2Dl"):
        assert parse("build", "log.json", "--profile", spelling).profile == spelling


def test_order_may_be_given_with_no_values():
    assert (
        parse("build", "log.json", "--order-representations").order_representations
        == []
    )


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit):
        parse("--version")
    assert "semlog" in capsys.readouterr().out


def test_build_writes_the_ontology_and_exits_zero(ocel, tmp_path):
    out = tmp_path / "onto"
    code = main(["build", str(ocel), "--output", str(out), "-q"])

    assert code == EXIT_OK
    assert [p.name for p in out.iterdir()] == ["pipeline.owl"]


def test_build_records_the_command_line_in_the_ontology(ocel, tmp_path):
    out = tmp_path / "onto"
    main(["build", str(ocel), "--output", str(out), "--case-concepts", "order", "-q"])

    command = run_record(out / "pipeline.owl")["buildCommand"]
    assert command.startswith("semlog build")
    assert "--case-concepts order" in command


def test_build_takes_its_name_from_the_config(ocel, tmp_path, write_config):
    config = write_config("hiring-el.toml", f'input = "{ocel}"\nprofile = "OWL2EL"\n')
    out = tmp_path / "onto"

    assert (
        main(["build", "--config", str(config), "--output", str(out), "-q"]) == EXIT_OK
    )
    record = run_record(out / "hiring-el.owl")
    assert record["buildConfiguration"]["profile"] == "OWL2EL"
    assert record["buildConfigFile"] == str(config)


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
    record = run_record(out / "run.owl")
    assert record["buildConfiguration"]["profile"] == "OWL2DL"


def test_a_positional_input_beats_the_config_file(ocel, tmp_path, write_config):
    config = write_config("run.toml", 'input = "does/not/exist.json"\n')
    out = tmp_path / "onto"

    code = main(
        ["build", str(ocel), "--config", str(config), "--output", str(out), "-q"]
    )
    assert code == EXIT_OK


def test_dry_run_prints_the_plan_and_writes_nothing(ocel, tmp_path, capsys):
    out = tmp_path / "onto"
    code = main(["build", str(ocel), "--output", str(out), "--dry-run", "-q"])

    assert code == EXIT_OK
    assert "would write" in capsys.readouterr().out
    assert not out.exists()


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


def test_a_missing_input_exits_two(tmp_path, capsys):
    code = main(["build", str(tmp_path / "absent.json"), "-q"])
    assert code == EXIT_USAGE
    assert "input not found" in capsys.readouterr().err


def test_a_failed_build_exits_one(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "broken.jsonocel").write_text("not valid ocel")

    code = main(["build", str(folder), "--output", str(tmp_path / "onto"), "-q"])
    assert code == EXIT_FAILED


def test_inspect_prints_a_summary(ocel, capsys):
    assert main(["inspect", str(ocel), "-q"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "object types" in out
    assert "--case-concepts" in out


def test_inspect_accepts_several_inputs(ocel, capsys):
    main(["inspect", str(ocel), str(ocel), "-q"])
    assert capsys.readouterr().out.count("object types") == 2


@pytest.mark.reasoner
def test_reason_checks_a_built_ontology(ocel, tmp_path):
    out = tmp_path / "onto"
    main(["build", str(ocel), "--output", str(out), "-q"])

    assert main(["reason", str(out / "pipeline.owl"), "--check", "-q"]) == EXIT_OK
