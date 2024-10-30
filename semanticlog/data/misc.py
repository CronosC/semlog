import os
import json
import sys
import pm4py as pm
import pandas as pd
from pm4py.algo.filtering.log.variants import variants_filter


def get_weighted_variant_mapping(
    log_path,
    case_column="case:concept:name",
    activity_column="concept:name",
    time_column="time:timestamp",
):
    if log_path.endswith(".xes"):
        log = pm.read_xes(log_path)
    elif log_path.endswith(".csv"):
        log = pd.read_csv(case_concept_log_path)
        log = pm.format_dataframe(
            log,
            case_id=case_column,
            activity_key=activity_column,
            timestamp_key=time_column,
        )
        log = pm.convert_to_event_log(log)
    else:
        raise ValueError("Invalid file path!", log_path)
    
    variants = variants_filter.get_variants(log)
    
    return variants