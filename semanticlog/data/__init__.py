"""Reading, holding and preparing object-centric event data.

`OCEDStore` is the model everything else in this package operates on. Loading
fills it, cleaning makes its names safe to emit, and flattening projects it
back into the case-per-row shape that classical process mining expects.
"""

from semanticlog.data.binning import Binning, fit_dist_bins, fit_equal_bins
from semanticlog.data.cleaning import clean_oced_data, fix_duplicate_names
from semanticlog.data.flatten import get_log, iter_logs
from semanticlog.data.oced import OCEDStore

__all__ = [
    "Binning",
    "OCEDStore",
    "clean_oced_data",
    "fit_dist_bins",
    "fit_equal_bins",
    "fix_duplicate_names",
    "get_log",
    "iter_logs",
]
