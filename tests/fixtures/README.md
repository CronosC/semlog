# tests/fixtures

Committed input to the test suite. The three logs the tests build from are
assembled in [../support/stores.py](../support/stores.py), and this directory
holds the one that comes from a file.

## `pipeline.jsonocel`

Written once with `pm4py.write_ocel2_json` and committed. There is no generator
in the test suite.

Shape, which the assertions depend on:

- **Timestamps.** Day *N* is `2024-01-01 + (N-1) days`, one event per day.
- **Three activities.** `place_order`, `pay`, `ship_order`. `place_order` and `pay` share
  the attributes `amount` and `resource`. `ship_order` has `carrier` and `weight` instead.
  The per-activity nulls that fall out of that are dropped by `PM4PyLoader`'s per-type
  `dropna(axis=1, how="all")`.
- **Two object types.** `order` (attribute `status`) and `item` (attribute `weight`), with
  the same per-type null dropping.
- **E2O.** `place_order` and `pay` relate to `order`, `ship_order` to `item`.
- **O2O.** `contains`, from `order` to `item`.

`weight` is owned by both an event type and an object type, which produces a two-element
property domain.

Object ids are `ord_1`..`ord_3` and `item_1`..`item_4`. A test that hands the
build a DECLARE constraints table has to use those, since a build refuses
constraints whose traces it cannot find in the log.

[tests/data/test_fixture_log.py](../data/test_fixture_log.py) asserts this shape,
so a change to the file fails there first.

There are no committed DECLARE fixtures. The tests that need one write it into their own
`tmp_path`.
