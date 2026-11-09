"""Dispatch to a subcommand, and the exit code and message each outcome gets."""

from __future__ import annotations

import sys

import pytest

from semanticlog.cli import EXIT_FAILED, EXIT_INTERRUPTED, EXIT_OK, EXIT_USAGE, main

# `semanticlog.cli` re-exports `main`, so the module comes from sys.modules.
MAIN_MODULE = "semanticlog.cli.main"


def test_no_command_prints_help_and_exits_two(capsys):
    assert main([]) == EXIT_USAGE
    assert "COMMAND" in capsys.readouterr().out


def test_a_dry_run_prints_the_plan_and_its_warnings_and_writes_nothing(
    ocel, tmp_path, capsys, caplog
):
    out = tmp_path / "onto"

    code = main(
        [
            "build",
            str(ocel),
            "--output",
            str(out),
            "--bins",
            "3",
            "--bin-labels",
            "a",
            "b",
            "c",
            "--dry-run",
        ]
    )

    assert code == EXIT_OK
    assert f"#   {out / 'pipeline.ofn'}" in capsys.readouterr().out
    assert "no effect" in caplog.text
    assert not out.exists()


def test_an_in_memory_build_prints_only_the_written_path(ocel, tmp_path, capsys):
    out = tmp_path / "onto"
    main(["build", str(ocel), "--output", str(out), "--no-streaming", "-q"])

    assert capsys.readouterr().out == f"{out / 'pipeline.owl'}\n"


def test_a_failed_build_exits_one(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "broken.jsonocel").write_text("not valid ocel")

    assert (
        main(["build", str(folder), "--output", str(tmp_path / "onto"), "-q"])
        == EXIT_FAILED
    )


def test_an_unusable_configuration_exits_two_with_one_line(ocel, capsys):
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
    err = capsys.readouterr().err
    assert err.startswith("semlog build: error: The settings contradict each other")
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "flags",
    [
        ["--size-limit-mb", "-1"],
        ["--auto-case-concepts", "0"],
        ["--max-ef-pairs", "-3"],
        ["--iri", "https://x.org/a b"],
    ],
    ids=["size limit", "auto case concepts", "max ef pairs", "iri"],
)
def test_a_setting_out_of_range_exits_two_with_one_line_before_the_log_is_read(
    ocel, capsys, monkeypatch, flags
):
    def never(_cfg):
        raise AssertionError("build_all ran")

    monkeypatch.setattr(sys.modules[MAIN_MODULE], "build_all", never)

    assert main(["build", str(ocel), *flags, "-q"]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert err.startswith(f"semlog build: error: {flags[0]}")
    assert err.count("\n") == 1


def test_an_unknown_profile_flag_names_the_flag(ocel, capsys):
    assert main(["build", str(ocel), "--profile", "foo", "-q"]) == EXIT_USAGE
    assert capsys.readouterr().err.startswith(
        "semlog build: error: --profile: unknown value 'foo'"
    )


def test_an_input_swallowed_by_a_list_flag_is_named_with_the_flag(ocel, capsys):
    code = main(["build", "--case-concepts", "order", str(ocel), "-q"])

    assert code == EXIT_USAGE
    err = capsys.readouterr().err
    assert f"Put {ocel} before --case-concepts" in err
    assert err.count("\n") == 1


def test_no_input_without_a_file_among_the_flags_asks_for_an_input(capsys):
    assert main(["build", "--case-concepts", "order", "-q"]) == EXIT_USAGE
    assert "No input given" in capsys.readouterr().err


def test_an_os_error_exits_two_with_one_line(ocel, tmp_path, capsys):
    taken = tmp_path / "onto"
    taken.write_text("not a directory", encoding="utf-8")

    assert main(["build", str(ocel), "--output", str(taken), "-q"]) == EXIT_USAGE
    assert capsys.readouterr().err.startswith("semlog build: error: ")


def test_an_error_that_is_not_a_semlog_error_propagates(ocel, monkeypatch):
    def burst(*_args, **_kwargs):
        raise ValueError("a plain value error")

    monkeypatch.setattr(sys.modules[MAIN_MODULE], "resolve_log_inputs", burst)

    with pytest.raises(ValueError, match="a plain value error"):
        main(["inspect", str(ocel), "-q"])


def test_an_interrupted_command_exits_130(ocel, capsys, monkeypatch):
    def interrupt(_cfg):
        raise KeyboardInterrupt

    monkeypatch.setattr(sys.modules[MAIN_MODULE], "build_all", interrupt)

    assert main(["build", str(ocel), "-q"]) == EXIT_INTERRUPTED
    assert capsys.readouterr().err.strip() == "semlog build: interrupted"
