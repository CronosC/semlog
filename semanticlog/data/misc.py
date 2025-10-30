from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.variants.pandas.get import get_variants_count
from semanticlog.data.oced import OCEDStore
import pandas as pd
import numpy as np
import pandas as pd
from scipy import stats

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

def quantize(data:OCEDStore) -> OCEDStore:
    for object_type in data.get_object_types():
        object_dict = data.get_object(object_type)
        df = object_dict["data"]
        id_col = object_dict["id"]
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if col not in [id_col]:
                    df[col] = quantize_series(df[col])
        object_dict["data"] = df
        data._objects[object_type] = object_dict
        
    for activity in data.get_activities():
        activity_dict = data.get_event(activity)
        df = activity_dict["data"]
        time_col = activity_dict["time"]
        id_col = activity_dict["id"]
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if col not in [time_col, id_col]:
                    df[col] = quantize_series(df[col])
        activity_dict["data"] = df
        data._events[activity] = activity_dict      
    return data
                
def quantize_series(series, bins=5, labels=["very_low", "low", "average", "high", "very_high"], min_values=10):
    if series.nunique() <= max(bins, min_values):
        #print("few values, direct categorisation")
        return series.astype(str)
    best_fit = get_best_fit_distribution(series)[1]
    if best_fit == "norm":
        return gaussian_quantize(series, bins=bins, labels=labels)
    if best_fit == "lognorm":
        return lognorm_quantize(series, bins=bins, labels=labels)
    if best_fit == "expon":
        return expon_quantize(series, bins=bins, labels=labels)
    if best_fit == "gamma":
        return gamma_quantize(series, bins=bins, labels=labels)
    else:
        return simple_quantize(series, bins=bins, labels=labels)

def get_best_fit_distribution(x, candidates=None, p=0.0):
    x = x.dropna().to_numpy()
    if candidates is None:
        candidates = ['norm','lognorm','expon','gamma']
    best = None
    for dist_name in candidates:
        dist = getattr(stats, dist_name)
        try:
            # fit with safe warnings off
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                if not (len(x) == 0 or (x <= 0).any() and dist_name in ["lognorm", "gamma"]):
                    params = dist.fit(x)
                    # compute log-likelihood
                    loglik = np.sum(dist.logpdf(x, *params))
                    aic = 2 * len(params) - 2 * loglik  # Akaike Information Criterion
                    print(f"{dist_name} fits with aic {aic}")
                    # compute p-value via KS test
                    ks_stat, p_val = stats.kstest(x, dist_name, args=params)

                    if best is None or aic < best[0]:
                        best = (aic, dist_name, params, p_val)
        except Exception:
            continue
    #print(f"best is {best[1]} with p val of {best[3]}")
    if best[3] < p:
        best = (None, None, None, None)
        #print("p val too low, no well fitting distribution found")
    return best  # (aic, name, params, pval)

def simple_quantize(series, bins=3, labels=["Low", "Average", "High"]):
    unique_vals = np.unique(series.dropna())
    if len(unique_vals) < bins:
        bins = len(unique_vals)
    quantiles = pd.qcut(unique_vals, q=bins, labels=labels, duplicates="drop")
    mapping = dict(zip(unique_vals, quantiles))
    return series.map(mapping)

def gaussian_quantize(series, bins=3, labels=["Low", "Average", "High"]):
    x = series.dropna()
    if len(x) == 0:
        return pd.Series([np.nan]*len(series), index=series.index)
    mu, sigma = stats.norm.fit(x)
    
    z_edges = np.linspace(-3, 3, bins + 1)
    z_edges[0], z_edges[-1] = -np.inf, np.inf
    z = (series - mu) / sigma
    
    return pd.cut(z, bins=z_edges, labels=labels)



def lognorm_quantize(series, bins=3, labels=["Low", "Average", "High"]):
    x = series.dropna()
    if len(x) == 0 or (x <= 0).all():
        return pd.Series([np.nan]*len(series), index=series.index)

    shape, loc, scale = stats.lognorm.fit(x, floc=0)
    cdf = stats.lognorm.cdf(series, shape, loc, scale)
    z = stats.norm.ppf(cdf)   # convert to “normal equivalent deviate”
    z_edges = np.linspace(-3, 3, bins + 1)
    z_edges[0], z_edges[-1] = -np.inf, np.inf
    return pd.cut(z, bins=z_edges, labels=labels)


def expon_quantize(series, bins=3, labels=["Low", "Average", "High"]):
    x = series.dropna()
    if len(x) == 0 or (x <= 0).all():
        return pd.Series([np.nan]*len(series), index=series.index)

    loc, scale = stats.expon.fit(x)
    cdf = stats.expon.cdf(series, loc, scale)
    z = stats.norm.ppf(cdf)
    z_edges = np.linspace(-3, 3, bins + 1)
    z_edges[0], z_edges[-1] = -np.inf, np.inf
    return pd.cut(z, bins=z_edges, labels=labels)


def gamma_quantize(series, bins=3, labels=["Low", "Average", "High"]):
    x = series.dropna()
    if len(x) == 0 or (x <= 0).all():
        return pd.Series([np.nan]*len(series), index=series.index)

    a, loc, scale = stats.gamma.fit(x, floc=0)
    cdf = stats.gamma.cdf(series, a, loc, scale)
    z = stats.norm.ppf(cdf)
    z_edges = np.linspace(-3, 3, bins + 1)
    z_edges[0], z_edges[-1] = -np.inf, np.inf
    return pd.cut(z, bins=z_edges, labels=labels)

    