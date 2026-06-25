from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter

from semanticlog.data.oced import OCEDStore


@dataclass
class DeclareResult:
    """Postprocessed DECLARE constraint results from mining or enumeration.

    Attributes:
        constraints: DataFrame of constraint satisfaction states per trace.
            Contains a ``case_concept`` column identifying the case concept.
        case_concepts: Object types used as case concepts during mining.
    """

    constraints: pd.DataFrame
    case_concepts: list[str]

    def write(self, path: str) -> None:
        """Write constraints to a CSV file."""
        self.constraints.to_csv(path, index=False)

    @classmethod
    def read(cls, path: str) -> DeclareResult:
        """Read a DeclareResult from a CSV file previously written by write().

        case_concepts are inferred from the ``case_concept`` column in the CSV.
        """
        constraints = pd.read_csv(path)
        case_concepts = constraints["case_concept"].unique().tolist()
        return cls(constraints=constraints, case_concepts=case_concepts)

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict for embedding in settings."""
        return {
            "constraints": self.constraints.to_dict("records"),
            "case_concepts": self.case_concepts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DeclareResult:
        """Restore from a dict produced by to_dict()."""
        return cls(
            constraints=pd.DataFrame(d["constraints"]),
            case_concepts=d["case_concepts"],
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

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
if hasattr(pd.errors, "SettingWithCopyWarning"):
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


class DeclareEnricher:
    """Mines DECLARE constraints from an OCEDStore using declare4py.

    Requires the ``[declare]`` extra (``declare4py``) and the Lydia LTLf/LDLf
    solver to be installed separately.  Import will succeed without them; the
    constructor raises ``ImportError`` at runtime if Declare4Py is unavailable.
    """

    def __init__(self, data: OCEDStore):
        """Initialise the enricher with the event store to mine from.

        Args:
            data: Cleaned :class:`~semanticlog.data.oced.OCEDStore`.

        Raises:
            ImportError: If Declare4Py or its Lydia backend is not installed.
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
        """Check a fixed set of DECLARE constraints against every trace exhaustively.

        Unlike discovery, this method checks all constraint/activity combinations
        in ``constraint_set`` without a support threshold.

        Args:
            case_concepts: Object types to use as case concepts.
            constraint_set: DECLARE constraint template names to evaluate
                (e.g. ``["Response", "Precedence", "Init"]``).
            vacuous_is_considered: Whether vacuously satisfied constraints count
                as satisfied.

        Returns:
            :class:`DeclareResult` with per-trace satisfaction states.
        """
        results = {}
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
        """Build a Declare4Py model string for all activity/constraint combinations.

        Args:
            log: A D4PyEventLog (as returned by :meth:`get_log`).
            constraint_set: Constraint template names to include. Unary templates
                (``Existence1``, ``Init``, ``End``, etc.) generate one clause per
                activity; binary templates generate one clause per ordered pair.

        Returns:
            Declare4Py model string ready for ``DeclareModel.parse_from_string``.
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
        """Discover DECLARE constraints meeting a minimum support threshold.

        Uses ``DeclareMiner`` to find constraints, then conformance-checks all
        traces against the discovered model.

        Args:
            case_concepts: Object types to use as case concepts.
            constraint_set: Constraint template names to consider during discovery.
                Pass ``None`` or ``[]`` to let the miner use its defaults.
            vacuous_is_considered: Whether vacuously satisfied constraints count
                as satisfied.
            constraint_support_threshold: Minimum fraction of traces a constraint
                must hold in to be included (0–1).
            itemset_support_threshold: Minimum itemset support for the underlying
                frequent-pattern step (0–1).

        Returns:
            :class:`DeclareResult` with per-trace satisfaction states for all
            discovered constraints.
        """
        if constraint_set is None:
            constraint_set = []
        results = {}
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
        """Normalise raw Declare4Py result DataFrames and concatenate them.

        Strips pipe/whitespace artifacts from column names, sanitises bracket
        notation to double-underscores, casts 0/1 columns to bool, and drops
        constant-valued boolean columns (no discrimination power).

        Args:
            results: Dict mapping case-concept name to a raw Declare4Py result DataFrame.

        Returns:
            Concatenated DataFrame with a ``case_concept`` column added.
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
        """Convert a case concept's flat DataFrame into a D4PyEventLog.

        Flattens the OCEDStore to a pm4py DataFrame via ``get_log``, converts it
        to a pm4py EventLog, then wraps it in a ``D4PyEventLog`` for Declare4Py.

        Args:
            case_concept: Object type to use as the case concept.

        Returns:
            Tuple of ``(D4PyEventLog, list[str])`` where the list contains the
            trace IDs in the same order as the log.
        """
        log = self.data.get_log(case_concept)
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
