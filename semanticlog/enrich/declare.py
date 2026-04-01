import warnings

import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter

from semanticlog.data.oced import OCEDStore

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
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


class DeclareEnricher:

    def __init__(self, data: OCEDStore):
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
    ):
        results = {}
        for case_concept in case_concepts:
            log, case_ids = self.get_log(case_concept)
            model = DeclareModel().parse_from_string(
                self.construct_model_string(log, constraint_set=constraint_set)
            )
            checker = MPDeclareAnalyzer(
                log=log, declare_model=model, consider_vacuity=False
            )
            result_df = checker.run().get_metric(metric="state")
            result_df["trace_id"] = case_ids
            results[case_concept] = result_df

        return self.postprocess_results(results)

    def construct_model_string(
        self,
        log,
        constraint_set=[],
    ):
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
            model_string = model_string + "activity " + activity + "\n"

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
        constraint_set: list[str] = [],
        vacuous_is_considered: bool = False,
        constraint_support_threshold: float = 0.5,
        itemset_support_threshold: float = 0.5,
    ):
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
                log=log, declare_model=model, consider_vacuity=False
            )
            result_df = checker.run().get_metric(metric="state")
            result_df["trace_id"] = case_ids
            results[case_concept] = result_df

        return self.postprocess_results(results)

    def postprocess_results(self, results):
        for result in results:
            result_df = results[result]

            # drop attribute value remnants
            result_df.columns = [col.strip(" |") for col in result_df.columns]

            # replace brackets
            result_df.columns = [
                col.replace("[", "__")
                .replace("]", "__")
                .replace(", ", "__")
                .replace(" ", "")
                for col in result_df.columns
            ]

            # make bool
            for col in result_df.columns:
                if result_df[col].dropna().isin([0, 1]).all():
                    result_df[col] = result_df[col].astype(bool)

            # drop all bools where the result is only false or true
            result_df = result_df.loc[
                :, ~result_df.dtypes.eq(bool) | (result_df.nunique(dropna=False) > 1)
            ]

            results[result] = result_df

        results_df = pd.concat(
            [df.assign(case_concept=key) for key, df in results.items()],
            ignore_index=True,
        )
        return results_df

    def get_log(self, case_concept):
        # df log
        log = self.data.get_log(case_concept)
        # legacy pm4py log
        parameters = {
            log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"
        }
        log = log_converter.apply(
            log,
            variant=log_converter.Variants.TO_EVENT_LOG,
            parameters=parameters,
        )
        case_ids = [trace.attributes["concept:name"] for trace in log]
        # d4py log
        log = D4PyEventLog(log=log, case_name="case:concept:name")

        return log, case_ids
