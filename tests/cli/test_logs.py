"""Console and per-input file logging."""

import logging

import pytest

from semanticlog.cli import logs
from semanticlog.cli.logs import PACKAGE_LOGGER, FileLog, setup_console_logging


@pytest.fixture
def restore_logging():
    """Put the root and package loggers and the console handler back as the suite had them."""
    root = logging.getLogger()
    package = logging.getLogger(PACKAGE_LOGGER)
    handlers, root_level, package_level = list(root.handlers), root.level, package.level
    console = logs._console_handler
    yield
    root.handlers[:] = handlers
    # A later setup_console_logging must replace the restored handler.
    logs._console_handler = console
    root.setLevel(root_level)
    package.setLevel(package_level)


def test_verbose_raises_this_package_only(restore_logging):
    setup_console_logging(logging.DEBUG)

    assert logging.getLogger("semanticlog.cli.run").isEnabledFor(logging.DEBUG)
    assert not logging.getLogger("pm4py").isEnabledFor(logging.DEBUG)
    assert logging.getLogger("pm4py").isEnabledFor(logging.WARNING)


def test_console_setup_leaves_pre_existing_handlers_alone(restore_logging):
    root = logging.getLogger()
    theirs = logging.NullHandler()
    root.addHandler(theirs)
    setup_console_logging()

    assert theirs in root.handlers


def test_setting_up_the_console_twice_adds_one_handler(restore_logging):
    root = logging.getLogger()

    setup_console_logging()
    after_one = len(root.handlers)
    setup_console_logging()

    assert len(root.handlers) == after_one


def test_file_logging_is_disabled_without_a_directory():
    assert FileLog(None).switch_to("stem") is None
    assert FileLog("").switch_to("stem") is None


def test_a_file_log_keeps_only_the_newest_files_of_an_input(tmp_path):
    with FileLog(tmp_path, keep=2) as log:
        for day in range(1, 5):
            assert log.switch_to("stem") is not None
            (tmp_path / f"2000010{day}_000000_stem.log").write_text("")

    kept = sorted(p.name for p in tmp_path.glob("2000010*_stem.log"))
    assert kept == ["20000103_000000_stem.log", "20000104_000000_stem.log"]


def test_keeping_no_files_is_rejected():
    with pytest.raises(ValueError, match="keep must be at least 1"):
        FileLog("logs", keep=0)


def test_leaving_a_file_log_detaches_its_handler_and_closing_again_changes_nothing(
    tmp_path,
):
    root = logging.getLogger()
    before = len(root.handlers)
    with FileLog(tmp_path) as log:
        log.switch_to("stem")
        assert len(root.handlers) == before + 1

    log.close()

    assert len(root.handlers) == before
