# tests

`tests/<layer>/` mirrors `semanticlog/<layer>/`. Anything a test imports lives
under `tests/support/`, so no test module is ever imported by another.

```
conftest.py     the shared fixtures, and the JVM check that skips the reasoner tests
fixtures/       committed input: the one OCEL 2.0 file
support/        importable helpers, never collected
core/ data/ enrich/ onto/ cli/
```

## The three stores

Every test that needs data uses one of these, from
[support/stores.py](support/stores.py):

| Store | What it is | Reach for it when |
|---|---|---|
| `tiny` | three events of one activity on one order | the data is not what is under test |
| `pipeline` | `fixtures/pipeline.jsonocel`, loaded and cleaned | the test should run against a real OCEL file |
| `awkward` | twelve orders and four items, carrying every hard case | a hard case has to be reachable |

`awkward` is what the build configuration matrix builds from, so every
configuration meets a name cleaning rewrote, a missing value, two events at one
instant, a numeric column wide enough to bin and a value at the far end of it.

Fixtures hand out a copy, since a store is mutable.

## The build matrix

[support/settings.py](support/settings.py) names one build configuration per
combination worth covering. Adding an entry there adds a case to every test
that runs over the matrix, and a row to the shape table.

The `built` fixture builds each configuration once per session, through each
sink, and returns it read back as an
[`Emitted`](support/ontology.py) object. Tests read from that rather than
building again. `built_path` returns the file the same build wrote, for a test
that needs a reader other than `Emitted`. Both share one cache, so asking for
either builds the configuration once.

## Two ways to read a build

[support/ontology.py](support/ontology.py) reads a build as text and answers
what its axioms say. Either syntax comes back as the same `Emitted` object, so
a test asserts against the ontology rather than the syntax, and the two sinks
compare directly.

[support/owlapi.py](support/owlapi.py) answers what the OWL API makes of the
same file: which profile it is in, what it carries as an ontology annotation,
and what its signature holds against what it declares. Those are questions
about the structure a parser builds, which a reader working from the lines
cannot see. It needs a JVM, so a test using it carries the `reasoner` marker.

## What checks the output

| File | Question it answers |
|---|---|
| [onto/test_build_shape.py](onto/test_build_shape.py) | did what a configuration emits change? |
| [onto/test_build_invariants.py](onto/test_build_invariants.py) | is the result a usable ontology? |
| [onto/test_build_content.py](onto/test_build_content.py) | does it say the right thing about the log? |
| [onto/test_profiles.py](onto/test_profiles.py) | is a build in the OWL 2 profile its settings name? |
| [onto/test_sinks.py](onto/test_sinks.py) | do both sinks agree? |
| [onto/test_reason.py](onto/test_reason.py) | what does a reasoner make of a build? |
| [test_imports.py](test_imports.py) | does importing the package start anything heavy? |

`test_build_shape.py` holds the record of the product: entity counts per
configuration, plus the names that configuration exists to produce and the ones
it must leave out. A change to the emitter that moves a number fails there,
naming the configuration. Update the row and say in the commit message which
change moved it.

## Running

```bash
pytest -n auto                                  # everything
pytest -n auto -m "not reasoner and not slow"   # no JVM, no subprocess
```

Reasoner tests skip themselves when the machine has no JVM. `slow` marks the
tests that run the installed command as a real process.
