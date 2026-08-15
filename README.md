# semlog

`semlog` turns object-centric event data into OWL 2 ontologies. It reads an OCEL 2.0 event log, or a flat CSV or XES log, normalizes it into an internal data model, and writes a single ontology file. That file is `<name>.ofn` in OWL functional syntax by default, which a reasoner and the OWL API read directly. Set `streaming = false` for `<name>.owl` in the RDF/XML a triple store, a SPARQL endpoint or rdflib expects.

## Installation

Python 3.12 or newer is required. `semlog reason` and `--materialize` run the OWL API on Java, so those two also need a JRE or JDK 17 or newer on PATH or named by `JAVA_HOME`. Building an ontology needs no JVM.

```bash
uv sync --extra dev          # recommended
uv run pre-commit install
```

or with pip:

```bash
pip install -e ".[dev]"
```

### DECLARE support

Mining DECLARE constraints is optional and needs the `[declare]` extra plus the [Lydia](https://github.com/whitemech/lydia) LTLf/LDLf solver, which is not on PyPI. Lydia is built from source against CUDD 3.0.0, MONA 1.4 and Syft, which in turn need `cmake`, `build-essential`, `bison`, `flex`, `libboost-dev` and `graphviz-dev`. [Declare4Py](https://github.com/ivanDonadello/Declare4Py) documents the setup. Everything except `semlog mine-declare` works without it.

## Example

The repository ships one small OCEL log and one config, so this runs from a fresh clone with no extra data:

```bash
semlog inspect tests/fixtures/pipeline.jsonocel
semlog build --config examples/pipeline.toml
semlog reason examples/out/pipeline.ofn --check
```

The same build from Python, in the three steps the CLI runs:

```python
from semanticlog import (
    BuildInputs,
    EventOrderRepresentation,
    OntoBuildSettings,
    OWLProfile,
    build_log,
    choose_case_concepts,
    load_and_clean,
)

data = load_and_clean("tests/fixtures/pipeline.jsonocel")
case_concepts, _ = choose_case_concepts(data, None)

settings = OntoBuildSettings(
    profile=OWLProfile.OWL2DL,
    order_representations=[EventOrderRepresentation.DF],
)
result = build_log(data, settings, BuildInputs(case_concepts), "out/pipeline.ofn")
```

`load_and_clean` reads the log and makes its names safe to emit. It must run before any build: it sanitizes every name that will become part of an IRI, resolves collisions with the names the build reserves for itself, and renames apart two source names that would otherwise clean to one. Every rename is recorded in the backmap. Anything still sharing a name once the renames have run is refused.

`choose_case_concepts` takes the object types you name, following any that cleaning renamed, or picks the largest ones the log can be projected onto. Passing `None` is what the CLI does without `--case-concepts`.

`OntoBuildSettings` holds the choices a build is configured with, and one settings object describes a build of any log. `BuildInputs` holds what has to be resolved against the log in hand: the case concepts, the mined DECLARE constraints, and any annotations to apply.

`build_log` writes the file, streaming by default. Pass `streaming=False` for RDF/XML, which also leaves the ontology on `result.onto` to inspect or reason over. Either way the file appears at its destination only once the build is complete, so a failure leaves whatever was there before it untouched.

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
| `iri` | `https://oced#` | Base IRI the ontology puts its names in. A trailing `#` is added when missing. Two ontologies built under one IRI are one ontology to a triple store. |
| `profile` | `OWL2DL` | Which OWL 2 profile the output stays inside. `OWL2EL` omits the inverse and transitivity axioms that profile forbids, and writes a float as `xsd:decimal`, a boolean and a date as `xsd:string`. |
| `order_representations` | `df` | How event order is represented: `df`, `ef`, `ef_explicit`, `declare`. Several can be combined. `ef` leaves the pairs to a reasoner, which derives them over `df`, so it has to be combined with it. `ef_explicit` asserts every pair itself and needs nothing else. |
| `case_concepts` | auto | Object types whose instances act as cases, named as the source log spells them. `df`, `ef` and `ef_explicit` need at least one. A type that cleaning renamed is followed to its new name. `declare` takes its traces from the mined CSV. |
| `data_representations` | `values` | How numeric attributes appear: `values` keeps the number, `equal_bins` buckets by frequency, `dist_bins` fits a distribution first. Each attribute is cut once, over every value it takes anywhere in the log. A column with too few distinct values to bin keeps its numbers. |
| `inverse_representations` | none | `axiomatic` writes inverse property axioms (OWL2DL only), `implicit` materializes the inverse assertions instead. |
| `ocel_extensions` | none | `attribute_history` also emits the timestamped attribute changes of each object. |
| `bins`, `bin_labels` | `5`, `very_low` … `very_high` | Bin count and labels for the two binning modes. |
| `max_ef_pairs` | `50000000` | Ceiling on materialized eventually-follows pairs. Zero for no limit. |
| `size_limit_mb` | `0` | Skip inputs larger than this many MB. Zero for no limit. |
| `derive_annotations` | `false` | Emit `rdfs:label` and `rdfs:comment` derived from the original names in the log. |
| `materialize` | `false` | Run a reasoner over the result and save what it entails. Needs a JVM. |
| `streaming` | `true` | Write OWL functional syntax straight to disk as `<name>.ofn`, keeping no axiom. What the build then holds grows with the distinct attribute values and timestamps of the log rather than with its events, and the loaded log itself is the floor. Off builds the ontology in memory and writes `<name>.owl` in RDF/XML, which is what rdflib reads. |

`semlog build --help` documents every flag in full.

### Reading a flat log

A `.csv` or `.xes` log has one case notion and no objects, so the case column becomes an object type and every event relates to its case. Any column named in `object_cols` becomes another object type the same way, and the rest are attributes.

```bash
semlog inspect log.csv --case-col case_id --act-col activity --time-col timestamp
semlog build log.csv --case-col case_id --act-col activity --time-col timestamp \
    --object-cols resource --case-concepts case_id
```

`case_col`, `act_col` and `time_col` default to the XES column names, so an `.xes` file usually needs none of them. `object_cols` is empty by default. All four are TOML keys as well as flags.

## Commands

| Command | Purpose |
|---|---|
| `semlog build` | Build one or more ontologies from event logs. |
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

A config naming one input takes its name from the config, so `runs/hiring.toml` writes `hiring.ofn`. A config naming several names each output after its input. `examples/pipeline.toml` is a complete config that runs from a fresh clone.

Exit codes are `0` for success, `1` when a build failed or an ontology was inconsistent, and `2` when the configuration was unusable.

## Development

```bash
make install     # dependencies and git hooks
make check       # lint, type check, tests
make test
make test-fast   # without the JVM and subprocess tests
make example     # build examples/pipeline.toml end to end
```

The reasoner tests load HermiT through the OWL API and need a JVM. They skip themselves when there is no Java.

[CHANGELOG.md](CHANGELOG.md) records the releases that moved the shape of the emitted file.

## License

MIT. See [LICENSE](LICENSE).
