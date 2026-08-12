"""Reading, holding and preparing object-centric event data.

`OCEDStore` is the model everything else in this package operates on. Loading
fills it, cleaning makes its names safe to emit, and flattening projects it
back into the case-per-row shape that classical process mining expects.
"""

from semanticlog.data.binning import dist_bin_series, equal_bin_series
from semanticlog.data.cleaning import clean_oced_data, fix_duplicate_names
from semanticlog.data.flatten import get_log, get_logs, iter_logs
from semanticlog.data.oced import OCEDStore

__all__ = [
    "OCEDStore",
    "dist_bin_series",
    "clean_oced_data",
    "equal_bin_series",
    "fix_duplicate_names",
    "get_log",
    "get_logs",
    "iter_logs",
]
