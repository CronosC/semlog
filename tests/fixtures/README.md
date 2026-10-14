# tests/fixtures

`pipeline.jsonocel` is the one event log the test suite reads from a file. It was written with `pm4py.write_ocel2_json` and has no generator. The other test logs are assembled in [../support/stores.py](../support/stores.py).

To see what it holds, run from the repository root:

```bash
uv run semlog inspect tests/fixtures/pipeline.jsonocel
```

## Shape the tests depend on

- **Timestamps.** Day *N* is `2024-01-01 + (N-1) days`, one event per day.
- **Three activities.** `place_order`, `pay`, `ship_order`. `place_order` and `pay` share the attributes `amount` and `resource`. `ship_order` has `carrier` and `weight`. `PM4PyLoader` drops a column that is null for a whole type, so no activity carries the other pair's attributes.
- **Two object types.** `order` with attribute `status`, and `item` with attribute `weight`. The same per-type null dropping applies.
- **E2O.** `place_order` and `pay` relate to `order`, `ship_order` to `item`.
- **O2O.** `contains`, from `order` to `item`.
- **Shared attribute.** `weight` belongs to an event type and an object type, which gives its property a two-element domain.
- **Object ids.** `ord_1`..`ord_3` and `item_1`..`item_4`. A build rejects DECLARE constraints whose traces are not in the log, so a constraints table you hand a build must use these ids.

If you change the file, update [../output/test_fixture_log.py](../output/test_fixture_log.py), which asserts this shape.

No DECLARE fixtures are committed. A test that needs one writes it into its own `tmp_path`.
