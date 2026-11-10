# tests

The test suite for `semlog`. Each module with logic has one test file at the mirrored path: `semlog/data/cleaning/allocate.py` is tested in `tests/data/cleaning/test_allocate.py`, and a package `__init__.py` in a file named after its main function. Modules without logic of their own, such as `core/errors.py`, have no test file. Tests of a whole build live in `tests/output/`.

## Running

From the repository root:

```bash
uv run pytest -n auto
uv run pytest -n auto -m "not reasoner and not slow"
uv run pytest tests/data/test_flatten.py -k timestamp
```

The second command skips the tests that need a JVM (`reasoner`) and the ones that run a subprocess or take seconds (`slow`). Every test that starts the JVM carries the `reasoner` marker, and the marked tests skip themselves on a machine without a JVM.

Under `-n auto`, the reasoner tests run on one worker, because each worker that runs one starts its own JVM. `conftest.py` puts them in one `xdist_group`, and `--dist loadgroup` in `pyproject.toml` keeps a group on one worker.

Every warning is an error, apart from the two PM4Py warnings `pyproject.toml` filters.

## Layout

```
conftest.py   shared fixtures, and the JVM check and xdist group for the reasoner marker
fixtures/     the one committed OCEL 2.0 file, with a README on what it holds
support/      importable helpers, never collected
core/ data/ enrich/ onto/ cli/   one test file per module
output/       whole builds, read back and checked
test_*.py     checks over the whole package, listed below
```

The test files at the root check the package as a whole:

| File | What it checks |
|---|---|
| [test_layering.py](test_layering.py) | no layer imports a layer above it |
| [test_imports.py](test_imports.py) | the exported names, and which modules load the JVM bridge and PM4Py |
| [test_raises_docs.py](test_raises_docs.py) | every `Raises:` block names the classes its function raises |
| [test_benchmarks.py](test_benchmarks.py) | each script in `benchmarks/` runs at a tiny size |
| [test_support_functional.py](test_support_functional.py) | the functional-syntax reader in `support/` reads a file the OWL API rewrote |

[cli/test_entrypoint.py](cli/test_entrypoint.py) mirrors no module. It runs the console script and `python -m semlog.cli.main` as subprocesses.

## Stores

Every test that needs data uses one of three stores from [support/stores.py](support/stores.py). Each fixture returns a copy.

| Store | What it is | Use it when |
|---|---|---|
| `tiny` | three events of one activity on one order | the data is not under test |
| `pipeline` | `fixtures/pipeline.jsonocel`, loaded and cleaned | the test needs a real OCEL file |
| `awkward` | twelve orders and four items with every hard case | a hard case must be reachable |

The build matrix builds from `awkward`. Every configuration meets a name cleaning rewrote, a missing value, two events at one timestamp, a numeric column wide enough to bin and a value at its far end.

## Building and reading back

[support/settings.py](support/settings.py) names the build configurations. `MATRIX` covers every value of every enum and boolean setting at least once, and `test_build_invariants.py` checks that it does. [support/build.py](support/build.py) holds `emit`, which builds one configuration through either sink, and `emit_through`, which also returns the sink.

The `built` fixture builds a configuration once per session and returns it as an [`Emitted`](support/emitted.py) object. `built_path` returns the file the same build wrote, and `built_counts` what its sink counted.

`Emitted` holds what a build's axioms say. [support/functional.py](support/functional.py) reads functional syntax into it, and [support/rdf.py](support/rdf.py) reads RDF/XML. [support/owlapi.py](support/owlapi.py) asks the OWL API about the same file: its profile, its ontology annotations and its undeclared signature. Tests that use it need the `reasoner` marker.

## Whole-build tests

| File | What it checks |
|---|---|
| [output/test_build_shape.py](output/test_build_shape.py) | the entity counts and names each configuration emits |
| [output/test_build_invariants.py](output/test_build_invariants.py) | every build is a well-formed ontology |
| [output/test_build_content.py](output/test_build_content.py) | a build states the right facts about the log |
| [output/test_build_refusals.py](output/test_build_refusals.py) | the checks only a whole build makes |
| [output/test_profiles.py](output/test_profiles.py) | a build stays in the OWL 2 profile its settings name |
| [output/test_fixture_log.py](output/test_fixture_log.py) | the committed fixture has the shape the tests rely on |

When an emitter change moves a number in `test_build_shape.py`, the test names the configuration. Update the row and say in the commit message which change moved it.

## Adding a test

- Name it `test_<scenario>_<expected>` so the name states the rule. Test one behavior.
- Put it in the file of the lowest module that holds the rule. Write a whole-build test only when the rule is visible only in the output.
- Give every `pytest.raises` on a semlog error a `match=`.
- Parametrize only over inputs that take different code paths.
- Leave out tests of field access, enum members and log text that is not product text.
