"""Mining DECLARE constraints with Declare4Py.

A DECLARE constraint is a rule about the order of activities in a trace, such
as one activity always being followed by another. `DeclareEnricher` mines them
per case concept and returns a `DeclareResult`.
"""

from __future__ import annotations

import contextlib
import io
import logging
import re
import warnings
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.core.errors import ConfigError, DependencyError, InputError
from semanticlog.data.declare import (
    TEMPLATES,
    ConceptConstraints,
    DeclareResult,
    MiningSettings,
)
from semanticlog.data.readable import ReadableStore

logger = logging.getLogger(__name__)


def template_of(column: str) -> str:
    """Return the template name of a mined constraint column such as `Response[a, b] | | |`."""
    return str(column).split("[", 1)[0].strip()


def _keep_templates(constraints: pd.DataFrame, wanted: list[str]) -> pd.DataFrame:
    """Return a copy of the constraint columns whose template is in `wanted`."""
    keep = [c for c in constraints.columns if template_of(c) in wanted]
    return constraints[keep].copy()


_CONSTRAINT_COLUMN = re.compile(r"^\s*(?P<template>[^\[]+?)\s*\[(?P<args>[^\]]*)\]")


def _constraint_name(column: str) -> str:
    """Return the class name of a Declare4Py column such as `Response[a, b] | | |`."""
    match = _CONSTRAINT_COLUMN.match(str(column))
    if match is None:
        raise DependencyError(
            f"Declare4Py returned the column {column!r}, which names no "
            "constraint. Install the version semlog pins with 'uv sync --extra "
            "declare'."
        )
    activities = [a.strip() for a in match["args"].split(",")]
    return vocab.declare_constraint(match["template"], activities)


def concept_constraints(case_concept: str, frame: pd.DataFrame) -> ConceptConstraints:
    """Return the constraints of one case concept from Declare4Py's output.

    Args:
        case_concept: Object type whose instances are the traces.
        frame: One row per trace, with a `trace_id` column and one column per
            constraint holding 1 where the trace satisfies it.

    Raises:
        DependencyError: If a column names no constraint, which the installed
            Declare4Py version decides.
        InputError: If two constraints get the same class name.
    """
    trace_ids = frame["trace_id"].astype(str).tolist()
    satisfied: dict[str, frozenset[str]] = {}
    sources: dict[str, str] = {}
    for column in frame.columns:
        if column == "trace_id":
            continue
        name = _constraint_name(column)
        if name in sources:
            raise InputError(
                f"Two constraints mined over {case_concept!r} both get the class "
                f"name {name!r}: {sources[name]!r} and {column!r}. Their activity "
                "names join to the same text. Rename one of the activities in the "
                "log."
            )
        sources[name] = column
        holds = frame[column].eq(1).tolist()
        satisfied[name] = frozenset(
            t for t, h in zip(trace_ids, holds, strict=True) if h
        )
    return ConceptConstraints(case_concept, tuple(trace_ids), satisfied)


def drop_uninformative(
    concepts: Sequence[ConceptConstraints],
) -> tuple[ConceptConstraints, ...]:
    """Drop every constraint that holds for all traces it was checked against, or for none.

    A constraint checked for several case concepts is judged over the traces
    of all of them.
    """
    checked: Counter[str] = Counter()
    held: Counter[str] = Counter()
    for concept in concepts:
        for name, ids in concept.satisfied.items():
            checked[name] += len(concept.trace_ids)
            held[name] += len(ids)
    constant = {name for name in checked if held[name] in (0, checked[name])}
    return tuple(
        ConceptConstraints(
            c.case_concept,
            c.trace_ids,
            {n: ids for n, ids in c.satisfied.items() if n not in constant},
        )
        for c in concepts
    )


def check_templates(names: list[str]) -> None:
    """Raise unless every name is a template Declare4Py knows.

    Raises:
        ConfigError: Naming the ones that are not, and what is available.
    """
    unknown = [name for name in names if name not in TEMPLATES]
    if unknown:
        raise ConfigError(
            f"Unknown DECLARE template(s): {', '.join(map(repr, unknown))}. "
            f"Choose from: {', '.join(sorted(TEMPLATES))}."
        )


@contextmanager
def _keep_logging_config() -> Iterator[None]:
    """Restore the root logger's handlers and level after the body runs.

    Declare4Py calls `logging.basicConfig` on import.
    """
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    try:
        yield
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


with _keep_logging_config():
    try:
        from Declare4Py.D4PyEventLog import D4PyEventLog
        from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import (
            MPDeclareAnalyzer,
        )
        from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner

        _DECLARE4PY_AVAILABLE = True
    except ImportError:
        _DECLARE4PY_AVAILABLE = False


class _InfoLines(io.TextIOBase):
    """Text stream that logs each line written to it at INFO."""

    def __init__(self) -> None:
        self._pending = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        *lines, self._pending = (self._pending + text).split("\n")
        for line in lines:
            # Declare4Py ends its progress lines in an ellipsis.
            message = line.strip().rstrip(" .")
            if message:
                logger.info("%s", message)
        return len(text)

    def flush(self) -> None:
        if self._pending:
            self.write("\n")


@contextmanager
def _quiet_declare4py() -> Iterator[None]:
    """Silence the warnings Declare4Py raises from inside pandas.

    What it prints is logged at INFO, so stdout keeps only what semlog prints.
    """
    with (
        warnings.catch_warnings(),
        contextlib.redirect_stdout(_InfoLines()) as lines,
    ):
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        if hasattr(pd.errors, "SettingWithCopyWarning"):
            warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
        try:
            yield
        finally:
            lines.flush()


class DeclareEnricher:
    """Mines DECLARE constraints from the traces of a store.

    The log is flattened once per case concept and each projection is mined
    separately.

    Args:
        data: Store to mine.

    Raises:
        DependencyError: If Declare4Py is not installed.
    """

    def __init__(self, data: ReadableStore) -> None:
        if not _DECLARE4PY_AVAILABLE:
            raise DependencyError(
                "semlog mine-declare needs Declare4Py and the Lydia solver. "
                "Install Declare4Py with 'uv sync --extra declare' and put the "
                "lydia binary on PATH, as the README's DECLARE support section "
                "describes."
            )
        self.data = data

    def get_constraints_via_discovery(
        self,
        case_concepts: list[str],
        constraint_set: list[str] | None = None,
        constraint_support_threshold: float = 0.5,
        itemset_support_threshold: float = 0.5,
    ) -> DeclareResult:
        """Discover the constraints the log supports, then check every trace.

        A constraint that holds only because its trigger never occurs does not
        count as satisfied.

        Args:
            case_concepts: Object types to project the log onto.
            constraint_set: Template names to keep from what the miner found.
                None keeps everything.
            constraint_support_threshold: Share of traces a constraint has to
                hold in to be proposed.
            itemset_support_threshold: Share of traces an activity set has to
                occur in to be considered.

        Returns:
            The result, with no origin recorded yet.

        Raises:
            ConfigError: If `case_concepts` is empty, or a name in
                `constraint_set` is not a Declare4Py template.
        """
        wanted = list(constraint_set or [])
        check_templates(wanted)
        if not case_concepts:
            raise ConfigError(
                "No case concepts to mine DECLARE constraints over. Name at "
                "least one object type. 'semlog inspect' lists the ones this "
                "log can be projected onto."
            )
        concepts: list[ConceptConstraints] = []
        with _quiet_declare4py():
            for case_concept in case_concepts:
                log, case_ids = self.get_log(case_concept)
                model = DeclareMiner(
                    log=log,
                    consider_vacuity=False,
                    min_support=constraint_support_threshold,
                    itemsets_support=itemset_support_threshold,
                    max_declare_cardinality=3,
                ).run()
                checker = MPDeclareAnalyzer(
                    log=log, declare_model=model, consider_vacuity=False
                )
                result_df = checker.run().get_metric(metric="state")
                if wanted:
                    result_df = _keep_templates(result_df, wanted)
                result_df["trace_id"] = case_ids
                concepts.append(concept_constraints(case_concept, result_df))

        mining = MiningSettings(
            "discovery",
            tuple(wanted) or None,
            constraint_support_threshold,
            itemset_support_threshold,
        )
        return DeclareResult(concepts=drop_uninformative(concepts), mining=mining)

    def get_log(self, case_concept: str) -> tuple[Any, list[Any]]:
        """Flatten the store onto one case concept, in the form Declare4Py reads.

        Args:
            case_concept: Object type whose instances become traces.

        Returns:
            The event log and the trace ids, in the order the log holds them.
        """
        from pm4py.objects.conversion.log import converter as log_converter
        from pm4py.util import constants as pm4py_constants
        from pm4py.util import xes_constants

        from semanticlog.data.flatten import get_log as flatten_log

        log = flatten_log(self.data, case_concept)
        parameters = {
            log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"
        }
        log = log_converter.apply(
            log,
            variant=log_converter.Variants.TO_EVENT_LOG,
            parameters=parameters,
        )
        # D4PyEventLog reads these two properties from the log. Conversion
        # from a DataFrame sets none.
        log.properties[pm4py_constants.PARAMETER_CONSTANT_TIMESTAMP_KEY] = (
            xes_constants.DEFAULT_TIMESTAMP_KEY
        )
        log.properties[pm4py_constants.PARAMETER_CONSTANT_ACTIVITY_KEY] = (
            xes_constants.DEFAULT_NAME_KEY
        )
        case_ids = [trace.attributes["concept:name"] for trace in log]
        log = D4PyEventLog(log=log, case_name="case:concept:name")

        return log, case_ids
