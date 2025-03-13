from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.variants.pandas.get import get_variants_count

def get_weighted_variant_mapping(
    log,
    case_column="case:concept:name",
    activity_column="concept:name",
    time_column="time:timestamp",
    event_column="event",
):
    """ Gets variant weights from a classical event log"""    
    log = log.sort_values(by=[case_column, time_column, activity_column, event_column])
    variant_weights = get_variants_count(log)

    
    return variant_weights