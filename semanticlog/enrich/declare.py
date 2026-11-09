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
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.core.errors import ConfigError, DependencyError, InputError
from semanticlog.data.declare import (
    BINARY_TEMPLATES,
    TEMPLATES,
    UNARY_TEMPLATES,
    ConceptConstraints,
    DeclareResult,
    MiningSettings,
)
from semanticlog.data.readable import ReadableStore

logger = logging.getLogger(__name__)

__all__ = ["BINARY_TEMPLATES", "TEMPLATES", "UNARY_TEMPLATES"]


def template_of(column: str) -> str:
    """Return the template name of a mined constraint column.

    A column is named `Response[a, b] | | |`. The template is the part before
    `[`.
    """
    return str(column).split("[", 1)[0].strip()


def _keep_templates(constraints: pd.DataFrame, wanted: list[str]) -> pd.DataFrame:
    """Drop the columns whose template is not in `wanted`.

    Args:
        constraints: One column per mined constraint.
        wanted: Template names to keep. Empty keeps everything.

    Returns:
        The frame with the unwanted columns removed.
    """
    if not wanted:
        return constraints
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
        from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import (  # noqa: E501
            MPDeclareAnalyzer,
        )
        from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner
        from Declare4Py.ProcessModels.DeclareModel import DeclareModel

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

    def get_constraints_via_enumeration(
        self,
        case_concepts: list[str],
        constraint_set: list[str],
        vacuous_is_considered: bool = False,
    ) -> DeclareResult:
        """Check a fixed set of constraint templates against every trace.

        Every template is instantiated for every activity, or ordered pair of
        activities, and then checked. The number of constraints is quadratic
        in the number of activities.

        Args:
            case_concepts: Object types to project the log onto.
            constraint_set: Template names to check, such as Response or Init.
            vacuous_is_considered: Count a constraint as satisfied when it
                holds only because its trigger never occurs.

        Returns:
            The result, with no origin recorded yet.

        Raises:
            ConfigError: If `case_concepts` is empty, or a name is not a
                Declare4Py template.
        """
        return self._check_each_trace(
            case_concepts,
            lambda log: DeclareModel().parse_from_string(
                self.construct_model_string(log, constraint_set=constraint_set)
            ),
            vacuous_is_considered,
            MiningSettings("enumeration", tuple(constraint_set)),
        )

    def _check_each_trace(
        self,
        case_concepts: list[str],
        model_for: Callable[[Any], Any],
        vacuous_is_considered: bool,
        mining: MiningSettings,
    ) -> DeclareResult:
        """Check one model per case concept against the traces of that concept.

        Args:
            case_concepts: Object types to project the log onto.
            model_for: Returns the DECLARE model to check one flat log against.
            vacuous_is_considered: Count a constraint as satisfied when it
                holds only because its trigger never occurs.
            mining: How the constraints are mined. Its templates, when set,
                are the ones kept of what the model produced.

        Returns:
            The result, with no origin recorded yet.

        Raises:
            ConfigError: If `case_concepts` is empty.
        """
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
                checker = MPDeclareAnalyzer(
                    log=log,
                    declare_model=model_for(log),
                    consider_vacuity=vacuous_is_considered,
                )
                result_df = checker.run().get_metric(metric="state")
                if mining.templates:
                    result_df = _keep_templates(result_df, list(mining.templates))
                result_df["trace_id"] = case_ids
                concepts.append(concept_constraints(case_concept, result_df))

        return DeclareResult(concepts=drop_uninformative(concepts), mining=mining)

    def construct_model_string(
        self,
        log: Any,
        constraint_set: list[str] | None = None,
    ) -> str:
        """Return the DECLARE model that enumeration checks against.

        Args:
            log: Flat log, read for the activities it contains.
            constraint_set: Template names to instantiate. Unary templates are
                instantiated once per activity, the rest once per ordered pair
                of distinct activities.

        Returns:
            The model, in the text form Declare4Py parses.

        Raises:
            ConfigError: If a name is not a Declare4Py template.
        """
        constraint_set = list(constraint_set or [])
        check_templates(constraint_set)
        activities = sorted(
            {event["concept:name"] for trace in log.get_log() for event in trace}
        )
        unary = [t for t in constraint_set if t in UNARY_TEMPLATES]
        binary = [t for t in constraint_set if t not in UNARY_TEMPLATES]

        lines = [f"activity {activity}" for activity in activities]
        for activity in activities:
            lines += [f"{template}[{activity}] | |" for template in unary]
            lines += [
                f"{template}[{activity}, {other}] | | |"
                for template in binary
                for other in activities
                if other != activity
            ]
        return "".join(f"{line}\n" for line in lines)

    def get_constraints_via_discovery(
        self,
        case_concepts: list[str],
        constraint_set: list[str] | None = None,
        vacuous_is_considered: bool = False,
        constraint_support_threshold: float = 0.5,
        itemset_support_threshold: float = 0.5,
    ) -> DeclareResult:
        """Discover the constraints the log supports, then check every trace.

        Args:
            case_concepts: Object types to project the log onto.
            constraint_set: Template names to keep from what the miner found.
                None keeps everything.
            vacuous_is_considered: Count a constraint as satisfied when it
                holds only because its trigger never occurs.
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

        def discovered(log: Any) -> Any:
            return DeclareMiner(
                log=log,
                consider_vacuity=vacuous_is_considered,
                min_support=constraint_support_threshold,
                itemsets_support=itemset_support_threshold,
                max_declare_cardinality=3,
            ).run()

        mining = MiningSettings(
            "discovery",
            tuple(wanted) or None,
            constraint_support_threshold,
            itemset_support_threshold,
        )
        return self._check_each_trace(
            case_concepts, discovered, vacuous_is_considered, mining
        )

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
