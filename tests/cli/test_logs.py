"""Console and per-input file logging."""

import logging

from semanticlog.cli.logs import FileLog, setup_console_logging


def test_console_setup_leaves_pre_existing_handlers_alone():
    """Importing the CLI must not discard a host process's logging setup."""
    root = logging.getLogger()
    theirs = logging.NullHandler()
    root.addHandler(theirs)
    try:
        setup_console_logging()
        assert theirs in root.handlers
    finally:
        root.removeHandler(theirs)


def test_console_setup_is_idempotent():
    root = logging.getLogger()
    setup_console_logging()
    before = len(root.handlers)
    setup_console_logging()
    assert len(root.handlers) == before


def test_file_logging_is_disabled_without_a_directory():
    assert FileLog(None).switch_to("stem") is None
    assert FileLog("").switch_to("stem") is None


def test_switching_writes_a_log_and_prunes_old_ones(tmp_path):
    with FileLog(tmp_path, keep=2) as log:
        for i in range(4):
            assert log.switch_to("stem") is not None
            # Timestamps have second resolution; make each name distinct.
            (tmp_path / f"2000010{i}_000000_stem.log").write_text("")

    assert len(list(tmp_path.glob("*_stem.log"))) <= 3


def test_close_is_idempotent(tmp_path):
    log = FileLog(tmp_path)
    log.switch_to("stem")
    log.close()
    log.close()


def test_the_context_manager_detaches_the_handler(tmp_path):
    root = logging.getLogger()
    before = len(root.handlers)
    with FileLog(tmp_path) as log:
        log.switch_to("stem")
        assert len(root.handlers) == before + 1
    assert len(root.handlers) == before
