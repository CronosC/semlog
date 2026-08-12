# Recorded output

`*.ofn` here is **output**, recorded from a build. `tests/fixtures/` is **input**.

Each file is what one entry of `tests/onto/support/configurations.py::CONFIGURATIONS`
emits through `StreamingSink`, sorted, with the version string and the
`buildConfiguration` settings JSON normalized out. A change to what a build emits
fails `test_golden.py`.

To re-baseline, say in the commit message which change altered the output:

```bash
SEMLOG_UPDATE_GOLDEN=1 pytest tests/onto/scenarios/test_golden.py
```

That run always ends red.
