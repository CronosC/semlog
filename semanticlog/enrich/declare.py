"""DECLARE constraints: mining them and carrying the result.

A DECLARE constraint is a rule about the order of activities in a trace, such
as one activity always being followed by another. `DeclareEnricher` mines them
with Declare4Py and the Lydia solver, and `DeclareResult` writes and reads them
as a CSV, which a build reads to type each trace by the constraints it
satisfies.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from semanticlog.core.errors import ConfigError, DependencyError, InputError
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import CONSTRAINT_INDEX_COLUMNS

logger = logging.getLogger(__name__)

META_SUFFIX = ".meta.json"

# The templates Declare4Py parses, from its DeclareModel grammar. Unary ones
# take one activity, the rest take an ordered pair.
UNARY_TEMPLATES: tuple[str, ...] = (
    "Absence1",
    "Absence2",
    "Absence3",
    "Existence1",
    "Existence2",
    "Existence3",
    "Exactly1",
    "Exactly2",
    "End",
    "Init",
)

BINARY_TEMPLATES: tuple[str, ...] = (
    "Choice",
    "Exclusive Choice",
    "Responded Existence",
    "Response",
    "Alternate Response",
    "Chain Response",
    "Precedence",
    "Alternate Precedence",
    "Chain Precedence",
    "Succession",
    "Alternate Succession",
    "Chain Succession",
    "Co-Existence",
    "Not Co-Existence",
    "Not Succession",
    "Not Chain Succession",
)

TEMPLATES: frozenset[str] = frozenset(UNARY_TEMPLATES) | frozenset(BINARY_TEMPLATES)


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


def _check_distinct_columns(
    renamed: list[str], sources: list[str], case_concept: str
) -> None:
    """Raise if two constraint names are equal after rewriting.

    Brackets and commas become underscores and spaces are removed.

    Args:
        renamed: The rewritten names, in order.
        sources: The names they were rewritten from, in the same order.
        case_concept: Case concept the frame belongs to, for the message.

    Raises:
        InputError: If two names collide. The message names the pair.
    """
    seen: dict[str, str] = {}
    for new, old in zip(renamed, sources, strict=True):
        if new in seen:
            raise InputError(
                f"Two constraints mined over {case_concept!r} are both called "
                f"{new!r} once their names are made safe to emit:\n"
                f"  {seen[new]!r}\n"
                f"  {old!r}\n\n"
                "Brackets and commas become underscores and spaces are "
                "removed. Clean the log before mining, which renames such "
                "activities apart."
            )
        seen[new] = old


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


@dataclass(frozen=True)
class DeclareResult:
    """Which traces satisfy which constraints, and where they came from.

    Attributes:
        constraints: One row per trace. `case_concept` and `trace_id` say
            which trace, and every other column is one constraint.
        case_concepts: The case concepts that were mined.
        source_log: Path of the log the constraints were mined from.
        source_sha256: Checksum of that log.
    """

    constraints: pd.DataFrame
    case_concepts: list[str]
    source_log: str | None = None
    source_sha256: str | None = None

    def write(self, path: str) -> None:
        """Write the constraints as CSV, with the origin in a JSON file beside it."""
        self.constraints.to_csv(path, index=False)
        Path(path + META_SUFFIX).write_text(
            json.dumps(
                {
                    "case_concepts": self.case_concepts,
                    "source_log": self.source_log,
                    "source_sha256": self.source_sha256,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    _UNNAMED_COLUMN = re.compile(r"^Unnamed: \d+$")

    INDEX_COLUMNS = CONSTRAINT_INDEX_COLUMNS

    @classmethod
    def read(cls, path: str) -> DeclareResult:
        """Read constraints from a CSV, with their origin if it is recorded.

        Unnamed index columns are dropped.

        Args:
            path: CSV to read.

        Returns:
            The constraints, with no origin recorded when the JSON file beside
            the CSV is missing.

        Raises:
            InputError: If the file does not have the shape of a constraints
                table, or the JSON file beside it is not a JSON object.
        """
        constraints = pd.read_csv(path)
        dropped = [c for c in constraints.columns if cls._UNNAMED_COLUMN.match(str(c))]
        if dropped:
            logger.info(
                "%s: dropping %s unnamed index column(s) (%s).",
                path,
                len(dropped),
                ", ".join(dropped),
            )
            constraints = constraints.drop(columns=dropped)

        missing = [c for c in cls.INDEX_COLUMNS if c not in constraints.columns]
        if missing:
            raise InputError(
                f"{path} is not a DECLARE constraints table: it has no "
                f"{' or '.join(repr(c) for c in missing)} column. Its columns are "
                f"{', '.join(map(str, constraints.columns))}. Mine constraints with "
                "'semlog mine-declare'."
            )

        case_concepts = constraints["case_concept"].unique().tolist()
        meta = cls._read_meta(path)
        return cls(
            constraints=constraints,
            case_concepts=case_concepts,
            source_log=meta.get("source_log"),
            source_sha256=meta.get("source_sha256"),
        )

    @staticmethod
    def _read_meta(path: str) -> dict:
        """Read the origin file beside a CSV, warning when there is none.

        Raises:
            InputError: If the file is not a JSON object.
        """
        meta_path = Path(path + META_SUFFIX)
        if not meta_path.is_file():
            logger.warning(
                "No %s beside %s. The build cannot check "
                "which log the constraints were mined from. Re-run "
                "'semlog mine-declare' to write it.",
                meta_path.name,
                path,
            )
            return {}
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InputError(
                f"{meta_path} is not readable as JSON ({exc}). It records which "
                "log the constraints beside it were mined from. Delete it to "
                "build without that check, or re-run 'semlog mine-declare'."
            ) from exc
        if not isinstance(meta, dict):
            raise InputError(
                f"{meta_path} holds {type(meta).__name__}, not an object with a "
                "'source_log' and a 'source_sha256'. Delete it to build without "
                "that check, or re-run 'semlog mine-declare'."
            )
        return meta


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


@contextmanager
def _quiet_declare4py() -> Iterator[None]:
    """Silence the warnings Declare4Py raises from inside pandas."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        if hasattr(pd.errors, "SettingWithCopyWarning"):
            warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
        yield


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
                "DeclareEnricher needs Declare4Py and its Lydia backend. Install "
                "Declare4Py and the Lydia LTLf/LDLf solver."
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
        )

    def _check_each_trace(
        self,
        case_concepts: list[str],
        model_for: Callable[[Any], Any],
        vacuous_is_considered: bool,
        keep_templates: list[str] | None = None,
    ) -> DeclareResult:
        """Check one model per case concept against the traces of that concept.

        Args:
            case_concepts: Object types to project the log onto.
            model_for: Returns the DECLARE model to check one flat log against.
            vacuous_is_considered: Count a constraint as satisfied when it
                holds only because its trigger never occurs.
            keep_templates: Template names to keep of what the model produced.
                None keeps everything.

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
        results: dict[str, pd.DataFrame] = {}
        with _quiet_declare4py():
            for case_concept in case_concepts:
                log, case_ids = self.get_log(case_concept)
                checker = MPDeclareAnalyzer(
                    log=log,
                    declare_model=model_for(log),
                    consider_vacuity=vacuous_is_considered,
                )
                result_df = checker.run().get_metric(metric="state")
                if keep_templates is not None:
                    result_df = _keep_templates(result_df, keep_templates)
                result_df["trace_id"] = case_ids
                results[case_concept] = result_df

        return DeclareResult(
            constraints=self.postprocess_results(results),
            case_concepts=case_concepts,
        )

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

        return self._check_each_trace(
            case_concepts, discovered, vacuous_is_considered, keep_templates=wanted
        )

    def postprocess_results(self, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Rewrite the miner output into class names and membership columns.

        Brackets and commas in column names become underscores and spaces are
        removed. An empty cell counts as not satisfied. A boolean column with
        one value across all traces of all case concepts is dropped.

        Args:
            results: One frame per case concept.

        Returns:
            The frames stacked, with a `case_concept` column added.

        Raises:
            InputError: If two constraint names are equal after rewriting.
        """
        for key, result_df in results.items():
            stripped = [col.strip(" |") for col in result_df.columns]
            renamed = [
                col.replace("[", "__")
                .replace("]", "__")
                .replace(", ", "__")
                .replace(" ", "")
                for col in stripped
            ]
            _check_distinct_columns(renamed, stripped, key)
            result_df.columns = renamed
            for col in result_df.columns:
                if col in CONSTRAINT_INDEX_COLUMNS:
                    continue
                if result_df[col].dropna().isin([0, 1]).all():
                    result_df[col] = result_df[col].eq(1)
            results[key] = result_df

        results_df = pd.concat(
            [df.assign(case_concept=key) for key, df in results.items()],
            ignore_index=True,
        )
        constant = [
            col
            for col in results_df.columns
            if col not in CONSTRAINT_INDEX_COLUMNS
            and results_df[col].dtype == bool
            and results_df[col].nunique(dropna=False) <= 1
        ]
        return results_df.drop(columns=constant)

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
