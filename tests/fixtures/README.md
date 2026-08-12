# tests/fixtures

Committed **input** to the test suite. Recorded **output** lives beside the test that
records it, in `tests/onto/scenarios/golden/`.

## `pipeline.jsonocel`

The one OCEL 2.0 log the integration and CLI tests build against. Written once with
`pm4py.write_ocel2_json` and committed. There is no generator in the test suite.

Shape, which is what the assertions depend on:

- **Timestamps.** Day *N* is `2024-01-01 + (N-1) days`, one event per day.
- **Three activities.** `place_order`, `pay`, `ship_order`. `place_order` and `pay` share
  the attributes `amount` and `resource`. `ship_order` has `carrier` and `weight` instead.
  The per-activity nulls that fall out of that are dropped by `PM4PyLoader`'s per-type
  `dropna(axis=1, how="all")`, which makes the attribute-owner union in the emitted
  ontology non-trivial.
- **Two object types.** `order` (attribute `status`) and `item` (attribute `weight`), with
  the same per-type null dropping.
- **E2O.** `place_order` and `pay` relate to `order`, `ship_order` to `item`.
- **O2O.** `contains`, from `order` to `item`.

`weight` is owned by both an event type and an object type, which produces a two-element
property domain.

There are no committed DECLARE fixtures. The tests that need one write it into their own
`tmp_path`.
