"""DECLARE constraints: mining them, and carrying the result.

A DECLARE constraint is a rule about the order of activities in a trace, such
as one activity always being followed by another. Mining them is a separate
step from building, because it needs the Lydia solver, which is not on PyPI
and has to be built from source. The mined constraints go to a CSV, and a
build reads that file to type each trace by the constraints it satisfies.

`DeclareResult` therefore works without any of that installed. `DeclareEnricher`
does not, and says so when it is constructed. What it needs to mine, including
the flattening, is imported when it runs rather than when this module loads.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from semanticlog.data.oced import OCEDStore

logger = logging.getLogger(__name__)

META_SUFFIX = ".meta.json"

# The templates declare4py parses, from its DeclareModel grammar. Unary ones
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

    A column arrives as `Response[a, b] | | |`, so the template is whatever
    comes before the activities.
    """
    return str(column).split("[", 1)[0].strip()


def _keep_templates(constraints: pd.DataFrame, wanted: list[str]) -> pd.DataFrame:
    """Drop the mined constraints of templates that were not asked for.

    Discovery proposes whatever the log supports, so restricting it to a few
    templates is done to what it found.

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


def _check_templates(names: list[str]) -> None:
    """Raise unless every name is a template declare4py knows.

    Raises:
        ValueError: Naming the ones that are not, and what is available.
    """
    unknown = [name for name in names if name not in TEMPLATES]
    if unknown:
        raise ValueError(
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
        source_sha256: Checksum of that log, which lets a build notice that
            the constraints belong to a different file.
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

    INDEX_COLUMNS = ("case_concept", "trace_id")

    @classmethod
    def read(cls, path: str) -> DeclareResult:
        """Read constraints from a CSV, with their origin if it is recorded.

        Unnamed index columns, which some tools write, are dropped. They would
        otherwise be read as constraints and become classes in the ontology.

        Args:
            path: CSV to read.

        Returns:
            The constraints, with no origin recorded when the JSON file beside
            the CSV is missing.

        Raises:
            ValueError: If the file does not have the shape of a constraints
                table.
        """
        constraints = pd.read_csv(path)
        dropped = [c for c in constraints.columns if cls._UNNAMED_COLUMN.match(str(c))]
        if dropped:
            logger.info(
                f"{path}: dropping {len(dropped)} unnamed index column(s) "
                f"({', '.join(dropped)}); they would become OWL classes."
            )
            constraints = constraints.drop(columns=dropped)

        missing = [c for c in cls.INDEX_COLUMNS if c not in constraints.columns]
        if missing:
            raise ValueError(
                f"{path} is not a DECLARE constraints table: it has no "
                f"{' or '.join(repr(c) for c in missing)} column. Its columns are "
                f"{', '.join(map(str, constraints.columns))}. Mine constraints with "
                "'semlog mine-declare', which writes this shape."
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
        """Read the origin file beside a CSV, warning when there is none."""
        meta_path = Path(path + META_SUFFIX)
        if not meta_path.is_file():
            logger.warning(
                f"No {meta_path.name} beside {path}: these constraints do not "
                "record which log they were mined from, so a build cannot check "
                "that they belong to it. Re-run 'semlog mine-declare' to get one."
            )
            return {}
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def to_dict(self) -> dict:
        """Return the result as plain values, for embedding in settings."""
        return {
            "constraints": self.constraints.to_dict("records"),
            "case_concepts": self.case_concepts,
            "source_log": self.source_log,
            "source_sha256": self.source_sha256,
        }


try:
    from Declare4Py.D4PyEventLog import D4PyEventLog
    from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import (
        MPDeclareAnalyzer,
    )
    from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner
    from Declare4Py.ProcessModels.DeclareModel import DeclareModel

    _DECLARE4PY_AVAILABLE = True
except ImportError:
    _DECLARE4PY_AVAILABLE = False


@contextmanager
def _quiet_declare4py():
    """Silence the warnings declare4py raises from inside pandas."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        if hasattr(pd.errors, "SettingWithCopyWarning"):
            warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
        yield


class DeclareEnricher:
    """Mines DECLARE constraints from the traces of a store.

    The log is flattened once per case concept, and each projection is mined
    on its own, so a constraint is always about the traces of one case notion.
    """

    def __init__(self, data: OCEDStore):
        """Prepare to mine one store.

        Args:
            data: Store to mine.

        Raises:
            ImportError: If declare4py or its solver is not installed.
        """
        if not _DECLARE4PY_AVAILABLE:
            raise ImportError(
                "Declare4Py and its Lydia backend are required for DeclareEnricher. "
                "Install declare4py and the Lydia LTLf/LDLf solver separately."
            )
        self.data = data

    def get_constraints_via_enumeration(
        self,
        case_concepts: list[str],
        constraint_set: list[str],
        vacuous_is_considered: bool = False,
    ) -> DeclareResult:
        """Check a fixed set of constraint templates against every trace.

        Every template is instantiated for every activity, or pair of
        activities, and then checked. The cost grows quickly with the number
        of activities.

        Args:
            case_concepts: Object types to project the log onto.
            constraint_set: Template names to check, such as Response or Init.
            vacuous_is_considered: Count a constraint as satisfied when it
                holds only because its trigger never occurs.

        Returns:
            The result, with no origin recorded yet.
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

        Enumeration and discovery differ only in where the model comes from,
        so they hand that in and share everything around it.

        Args:
            case_concepts: Object types to project the log onto.
            model_for: Returns the DECLARE model to check one flat log against.
            vacuous_is_considered: Count a constraint as satisfied when it
                holds only because its trigger never occurs.
            keep_templates: Template names to keep of what the model produced.
                None keeps everything.

        Returns:
            The result, with no origin recorded yet.
        """
        results = {}
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
        log,
        constraint_set=None,
    ) -> str:
        """Write the DECLARE model that enumeration checks against.

        Args:
            log: Flat log, read for the activities it contains.
            constraint_set: Template names to instantiate. Unary templates are
                instantiated once per activity, the rest once per ordered pair
                of distinct activities.

        Returns:
            The model, in the text form declare4py parses.

        Raises:
            ValueError: If a name is not a template declare4py knows. A
                misspelling would otherwise be written into the model as a
                binary template and fail inside the parser.
        """
        constraint_set = list(constraint_set or [])
        _check_templates(constraint_set)
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
        """Discover the constraints the log actually supports, then check them.

        Discovery proposes only constraints that hold often enough to be worth
        checking, which keeps this usable on a log with many activities. This
        is the one `semlog mine-declare` runs.

        Args:
            case_concepts: Object types to project the log onto.
            constraint_set: Template names to keep. The miner proposes
                whatever the log supports, so this is applied to what it
                found. None keeps everything.
            vacuous_is_considered: Count a constraint as satisfied when it
                holds only because its trigger never occurs.
            constraint_support_threshold: Share of traces a constraint has to
                hold in to be proposed.
            itemset_support_threshold: Share of traces an activity set has to
                occur in to be considered.

        Returns:
            The result, with no origin recorded yet.

        Raises:
            ValueError: If a name in `constraint_set` is not a template
                declare4py knows.
        """
        wanted = list(constraint_set or [])
        _check_templates(wanted)

        def discovered(log) -> Any:
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

    def postprocess_results(self, results: dict) -> pd.DataFrame:
        """Make the miner output fit to become class names and assertions.

        Constraint columns arrive with brackets, commas and spaces in their
        names, none of which can appear in an IRI, so they are rewritten with
        underscores. Columns that say the same thing about every trace are
        dropped, since a class every trace belongs to says nothing.

        Args:
            results: One frame per case concept.

        Returns:
            The frames stacked, with a `case_concept` column added.
        """
        for key, result_df in results.items():
            result_df.columns = [col.strip(" |") for col in result_df.columns]
            result_df.columns = [
                col.replace("[", "__")
                .replace("]", "__")
                .replace(", ", "__")
                .replace(" ", "")
                for col in result_df.columns
            ]
            for col in result_df.columns:
                if result_df[col].dropna().isin([0, 1]).all():
                    result_df[col] = result_df[col].astype(bool)
            result_df = result_df.loc[
                :, ~result_df.dtypes.eq(bool) | (result_df.nunique(dropna=False) > 1)
            ]
            results[key] = result_df

        results_df = pd.concat(
            [df.assign(case_concept=key) for key, df in results.items()],
            ignore_index=True,
        )
        return results_df

    def get_log(self, case_concept: str) -> tuple:
        """Flatten the store onto one case concept, in the form declare4py wants.

        Args:
            case_concept: Object type whose instances become traces.

        Returns:
            The event log and the trace ids, in the order the log holds them.
        """
        from pm4py.objects.conversion.log import converter as log_converter

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
        case_ids = [trace.attributes["concept:name"] for trace in log]
        log = D4PyEventLog(log=log, case_name="case:concept:name")

        return log, case_ids
