# semlog

`semlog` turns object-centric event data into OWL 2 ontologies. It reads an OCEL 2.0 event log, normalizes it into an internal data model, and writes a single `.owl` file that a reasoner, a triple store or a SPARQL endpoint can consume.

## Installation

Python 3.12 or newer is required.

```bash
uv sync --extra dev          # recommended
pre-commit install
```

or with pip:

```bash
pip install -e ".[dev]"
```

### DECLARE support

Mining DECLARE constraints is optional and needs the `[declare]` extra plus the [Lydia](https://github.com/whitemech/lydia) LTLf/LDLf solver. Lydia is not on PyPI and has to be built from source together with CUDD, MONA and Syft. Everything except `semlog mine-declare` works without it.

## Example

The repository ships one small OCEL log and one config, so this runs from a fresh clone with no extra data:

```bash
semlog inspect tests/fixtures/pipeline.jsonocel
semlog build --config examples/pipeline.toml
semlog reason examples/out/pipeline.owl --check
```

The same build from Python:

```python
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.loading.pm4py_loader import PM4PyLoader
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    EventOrderRepresentation,
    OntoBuildSettings,
    OWLProfile,
)

data = PM4PyLoader("tests/fixtures/pipeline.jsonocel", {}).get_oced()
data = clean_oced_data(data)

settings = OntoBuildSettings(
    profile=OWLProfile.OWL2DL,
    order_representations=[EventOrderRepresentation.DF],
    case_concepts=["order"],
)

result = OCEDOntoBuilder(DEFAULT_IRI, settings).build_ontology(data)
result.save("pipeline.owl")
```

`clean_oced_data` must run before any build. It sanitizes every name that will become part of an IRI and resolves collisions with the names the build reserves for itself.

## What a build produces

The ontology has a fixed upper schema of `Event`, `Object`, `Attribute` and `Time`, and everything from the log hangs below it:

- one class per activity and per object type, one individual per event and per object
- one class per attribute, with a named individual for each distinct value, so values can be reasoned over
- one individual per distinct timestamp, linked to its events through `observed_at`
- the relations of the log as sub-properties of `event_object_relation` and `object_object_relation`
- event order as `df_<case concept>` properties, optionally with eventually-follows and DECLARE constraint membership

The record of how the file was produced is annotated onto the ontology itself: the input checksum, the resolved settings, the command and the semlog version.

## Build settings

Every setting below is both a TOML key and a `--flag`. A flag always beats the config file.

| Setting | Default | What it changes |
|---|---|---|
| `profile` | `OWL2DL` | Which OWL 2 profile the output stays inside. `OWL2EL` omits the inverse and transitivity axioms that profile forbids. |
| `order_representations` | `df` | How event order is represented: `df`, `ef`, `ef_explicit`, `declare`. Several can be combined. |
| `case_concepts` | auto | Object types whose instances act as cases. Any event order needs at least one. |
| `data_representations` | `values` | How numeric attributes appear: `values` keeps the number, `equal_bins` buckets by frequency, `dist_bins` fits a distribution first. |
| `inverse_representations` | none | `axiomatic` writes inverse property axioms (OWL2DL only), `implicit` materializes the inverse assertions instead. |
| `ocel_extensions` | none | `attribute_history` also emits the timestamped attribute changes of each object. |
| `bins`, `bin_labels` | `5`, `very_low` … `very_high` | Bin count and labels for the two binning modes. |
| `size_limit_mb` | `0` | Skip inputs larger than this many MB. Zero for no limit. |
| `derive_annotations` | `false` | Emit `rdfs:label` and `rdfs:comment` derived from the original names in the log. |
| `materialize` | `false` | Run a reasoner over the result and save what it entails. |
| `streaming` | `false` | Write the ontology straight to disk at near constant memory, for logs too large to hold. Rules out `materialize`. |

`semlog build --help` documents every flag in full.

## Commands

| Command | Purpose |
|---|---|
| `semlog build` | Build one or more ontologies from OCEL files. |
| `semlog inspect` | Summarize a log: activities, object types, relations, time range, suggested case concepts. |
| `semlog mine-declare` | Mine DECLARE constraints into a CSV for `--order-representations declare`. Needs Lydia. |
| `semlog reason` | Materialize or consistency-check an ontology that already exists. |

Settings can live in a TOML run config, which keeps a build reproducible and lets several configurations of one dataset coexist:

```toml
# runs/hiring.toml
extends = "_base.toml"

input = "datasets/ocel/Hiring.xml"
case_concepts = ["Candidate", "JobRequisition"]
```

```bash
semlog build --config runs/hiring.toml
semlog build --config runs/hiring.toml --profile OWL2EL   # same run, EL instead
semlog build --config runs/hiring.toml --dry-run          # resolve and print, build nothing
```

Outputs are named after the config, so `runs/hiring.toml` writes `hiring.owl`. `examples/pipeline.toml` is a complete config that runs from a fresh clone.

Exit codes are `0` for success, `1` when a build failed or an ontology was inconsistent, and `2` when the configuration was unusable.

## Development

```bash
make install     # dependencies and git hooks
make check       # lint, type check, tests
make test
make example     # build examples/pipeline.toml end to end
```

The reasoner tests load HermiT through the OWL API and need a JVM. Without Java, run `pytest -m "not reasoner"`.

## License

MIT. See [LICENSE](LICENSE).
