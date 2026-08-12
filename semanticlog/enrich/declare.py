"""DECLARE constraints: mining them, and carrying the result.

A DECLARE constraint is a rule about the order of activities in a trace, such
as one activity always being followed by another. Mining them is a separate
step from building, because it needs the Lydia solver, which is not on PyPI
and has to be built from source. The mined constraints go to a CSV, and a
build reads that file to type each trace by the constraints it satisfies.

`DeclareResult` therefore works without any of that installed. `DeclareEnricher`
does not, and says so when it is constructed.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter

from semanticlog.data.flatten import get_log as flatten_log
from semanticlog.data.oced import OCEDStore

logger = logging.getLogger(__name__)

META_SUFFIX = ".meta.json"


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

    @classmethod
    def from_dict(cls, d: dict) -> DeclareResult:
        """Rebuild a result from the output of `to_dict`."""
        return cls(
            constraints=pd.DataFrame(d["constraints"]),
            case_concepts=d["case_concepts"],
            source_log=d.get("source_log"),
            source_sha256=d.get("source_sha256"),
        )


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
        results = {}
        with _quiet_declare4py():
            for case_concept in case_concepts:
                log, case_ids = self.get_log(case_concept)
                model = DeclareModel().parse_from_string(
                    self.construct_model_string(log, constraint_set=constraint_set)
                )
                checker = MPDeclareAnalyzer(
                    log=log, declare_model=model, consider_vacuity=vacuous_is_considered
                )
                result_df = checker.run().get_metric(metric="state")
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
        """
        if constraint_set is None:
            constraint_set = []
        model_string = ""
        activities = list(
            set(event["concept:name"] for trace in log.get_log() for event in trace)
        )
        unary_constraints = [
            "Absence1",
            "Existence1",
            "Absence2",
            "Existence2",
            "Absence3",
            "Existence3",
            "End",
            "Init",
        ]
        unary_constraints_to_include = [
            x for x in unary_constraints if x in constraint_set
        ]
        binary_constraint_to_include = [
            x for x in constraint_set if x not in unary_constraints
        ]

        for activity in activities:
            model_string += f"activity {activity}\n"

        for activity in activities:
            for unary_constraint in unary_constraints_to_include:
                model_string = f"{model_string}{unary_constraint}[{activity}] | |\n"

            for binary_constraint in binary_constraint_to_include:
                for activity2 in activities:
                    if activity != activity2:
                        model_string = f"{model_string}{binary_constraint}[{activity}, {activity2}] | | |\n"

        return model_string

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
        checking, which keeps this usable on a log with many activities.

        Args:
            case_concepts: Object types to project the log onto.
            constraint_set: Template names to restrict discovery to. None
                leaves the choice to the miner.
            vacuous_is_considered: Count a constraint as satisfied when it
                holds only because its trigger never occurs.
            constraint_support_threshold: Share of traces a constraint has to
                hold in to be proposed.
            itemset_support_threshold: Share of traces an activity set has to
                occur in to be considered.

        Returns:
            The result, with no origin recorded yet.
        """
        if constraint_set is None:
            constraint_set = []
        results = {}
        with _quiet_declare4py():
            for case_concept in case_concepts:
                log, case_ids = self.get_log(case_concept)
                discovery = DeclareMiner(
                    log=log,
                    consider_vacuity=vacuous_is_considered,
                    min_support=constraint_support_threshold,
                    itemsets_support=itemset_support_threshold,
                    max_declare_cardinality=3,
                )
                model = discovery.run()
                checker = MPDeclareAnalyzer(
                    log=log, declare_model=model, consider_vacuity=vacuous_is_considered
                )
                result_df = checker.run().get_metric(metric="state")
                result_df["trace_id"] = case_ids
                results[case_concept] = result_df

        return DeclareResult(
            constraints=self.postprocess_results(results),
            case_concepts=case_concepts,
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
