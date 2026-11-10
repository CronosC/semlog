"""Reading, holding and preparing object-centric event data.

`oced.OCEDStore` is the model everything else in this package operates on.
Loading fills it, cleaning makes its names safe to emit, and flattening
projects it back into the case-per-row shape that classical process mining
expects.
"""
