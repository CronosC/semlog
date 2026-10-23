# semlog

`semlog` turns an OCEL 2.0 event log, or a flat CSV or XES log, into one OWL 2 ontology file. The file is `<name>.ofn` in OWL functional syntax by default, or `<name>.owl` in RDF/XML with `--no-streaming`.

## Installation

You need Python 3.12 or newer. `semlog reason` and `--materialize` also need Java 17 or newer on `PATH` or in `JAVA_HOME`. Building an ontology needs no Java.

The package is not on PyPI. Clone the repository and install it with [uv](https://docs.astral.sh/uv/):

```bash
git clone git@github.com:CronosC/semlog.git
cd semlog
uv sync
uv run semlog --version
```

With pip instead, run `pip install .` in a virtual environment and drop the `uv run` prefix from the commands below.

## Running it

The repository ships one small log and one config. From a fresh clone, run:

```bash
uv run semlog inspect --config examples/pipeline.toml
uv run semlog build --config examples/pipeline.toml
uv run semlog reason examples/out/pipeline.ofn --check
```

`inspect` prints the activities, object types, relations and time range of the log. `build` writes `examples/out/pipeline.ofn` and prints its path. `reason --check` prints whether the ontology is consistent.

To build a log with flags, name the object types whose instances are the cases and how event order is written:

```bash
uv run semlog build tests/fixtures/pipeline.jsonocel --case-concepts order item --order-representations df
```

Without `--case-concepts`, the three most common object types are chosen. Without `--order-representations`, no event order is written. `uv run semlog inspect tests/fixtures/pipeline.jsonocel` lists the type names.

| Command | Purpose |
|---|---|
| `semlog build` | Build one ontology per event log. |
| `semlog inspect` | Summarize a log: activities, object types, relations, time range, and the activity and object type names cleaning changed. |
| `semlog mine-declare` | Mine DECLARE constraints into a JSON file for `--order-representations declare`. Needs Lydia. |
| `semlog reason` | Materialize or consistency-check an existing ontology. |

`semlog <command> --help` lists every flag.

### Config files

Every setting is a TOML key and a flag. `input` is positional on the command line. `reason` spells the backend flag `--backend`. A flag wins over the file.

Keep your own configs in `runs/`, which git ignores:

```bash
mkdir -p runs
```

A config can extend another one. Create `runs/el.toml` with:

```toml
extends = "../examples/pipeline.toml"
profile = "OWL2EL"
```

Then build it, or print what it resolves to:

```bash
uv run semlog build --config runs/el.toml
uv run semlog build --config runs/el.toml --dry-run
```

`extends` resolves relative to the config file. Other relative paths resolve against the current directory. A config with one input names the output after the config, so `runs/el.toml` writes `examples/out/el.ofn`. A config with several inputs names each output after its input. A build warns before it overwrites an ontology built from different settings.

`build`, `inspect` and `mine-declare` all take `--config` and read `input`, `ocel_extensions` and the column keys from it. `build` and `mine-declare` also read `case_concepts` and `auto_case_concepts`.

`--dry-run` prints the resolved run as TOML and builds nothing. To keep a run you put together with flags, redirect it to a file:

```bash
uv run semlog build tests/fixtures/pipeline.jsonocel --case-concepts order --profile OWL2EL --order-representations df ef_explicit --dry-run > runs/el-explicit.toml
uv run semlog build --config runs/el-explicit.toml
```

### Mining DECLARE constraints

`mine-declare` needs Lydia, see [DECLARE support](#declare-support). It writes to `--output`, else to the config's `declare_file`, else to `<input stem>.declare.json` in the current directory. The file holds, per case concept, the traces checked and the traces that satisfy each constraint, with the templates and thresholds it was mined with. Pass it to a build with `--declare-file`:

```bash
uv run semlog mine-declare tests/fixtures/pipeline.jsonocel --case-concepts order --output runs/pipeline.declare.json
uv run semlog build tests/fixtures/pipeline.jsonocel --case-concepts order --order-representations df declare --declare-file runs/pipeline.declare.json
```

`mine-declare --config` and `build --config` read the same file. It can hold the mining settings `constraints`, `constraint_support` and `itemset_support` beside the build settings.

### Output and exit codes

The paths a build wrote and the verdict of `reason --check` go to stdout. Progress and problems go to stderr, and `-q` limits stderr to warnings and errors. Exit codes are `0` for success, `1` when a build failed, a log could not be read or an ontology was inconsistent, `2` when the configuration was unusable or `mine-declare` or `reason` could not read its input, and `130` when the command was interrupted.

### Reading a flat log

A `.csv` or `.xes` log has one case notion and no objects. The case column becomes an object type and every event relates to its case. Each column named in `object_cols` becomes another object type the same way. A column named in `object_attribute_cols` becomes an attribute of the case, or of the first object column within whose objects it does not vary. The other columns become attributes of the events.

```bash
printf 'case_id,activity,timestamp,resource\nc1,register,2024-01-01 09:00,ann\nc1,approve,2024-01-02 10:00,bo\n' > runs/claims.csv
uv run semlog inspect runs/claims.csv --case-col case_id --act-col activity --time-col timestamp
uv run semlog build runs/claims.csv --case-col case_id --act-col activity --time-col timestamp --object-cols resource --case-concepts case_id --order-representations df
```

A timestamp without a time zone is read as UTC. `case_col`, `act_col` and `time_col` default to the XES column names, so an `.xes` file needs none of them unless it renames those columns. `object_cols` and `object_attribute_cols` are empty by default. All five are TOML keys and flags.

## What a build produces

The ontology has a fixed upper schema of `Event`, `Object`, `Attribute` and `Time`, and the classes and properties taken from the log are placed below it:

- one class per activity and per object type, one individual per event and per object
- one class per attribute name, below `EventAttribute`, `ObjectAttribute` or `Attribute` by the kinds of type that carry it, with a named individual for each distinct value carrying its literal through `has_value_<attribute>`. With `attribute_representations = ["literals"]` the event or object carries the literal itself instead
- one individual per distinct timestamp, linked to its events through `observed_at`
- the relations of the log as sub-properties of `event_object_relation` and `object_object_relation`
- event order as `df_<case concept>` properties, optionally with eventually-follows and DECLARE constraint membership

An OCEL attribute keeps the type the log declares for it: `string`, `integer`, `float`, `boolean` or `time`. A value that does not parse as its declared type is read as missing, with a warning. Undeclared attributes, and every column of a CSV or XES log, are typed by their content.

The record of how the file was produced is annotated onto the ontology itself: the input checksum, the resolved settings, the command and the semlog version.

## Build settings

| Setting | Default | What it changes |
|---|---|---|
| `output` | the current directory | Directory the ontology and any other artifact of the run go into. |
| `name` | the config or input name | Stem of every file the run writes. |
| `iri` | `https://example.org/oced/<stem>/` | Base IRI of every name from the log, with `<stem>` the artifact stem. A trailing `/` is added when missing. Classes and properties sit directly under it. Individuals sit under one path per kind below it: `event/`, `object/`, `value/`, `time/` and `change/`. Schema names are under `https://example.org/oced#`. |
| `tbox_iri` | `https://example.org/oced#` | Namespace of the schema classes and properties, such as `Event` and `observed_at`. Ends in `#` or `/`. |
| `profile` | `OWL2DL` | Which OWL 2 profile the output stays inside. `OWL2EL` rejects `ef` and `axiomatic` inverses, drops a domain or range that is a union of classes, and writes a float as `xsd:decimal` and a boolean or date as `xsd:string`. |
| `order_representations` | none | How event order is written: `df`, `ef`, `ef_explicit`, `declare`. Several can be combined. `ef` declares eventually-follows transitive over `df` and leaves the pairs to a reasoner, so it needs `df` beside it and `OWL2DL`. `ef_explicit` asserts every pair itself. When an event is in two traces of one case concept, `ef` also derives pairs across those traces and the build warns. `ef_explicit` does not. Events of one trace with the same timestamp are ordered by event id. |
| `case_concepts` | the 3 most common object types | Object types whose instances act as cases, spelled as the source log spells them. `df`, `ef` and `ef_explicit` need at least one. `declare` takes its traces from the mined file. |
| `auto_case_concepts` | `3` | How many object types are chosen when `case_concepts` is not set. |
| `attribute_representations` | `individuals` | How attribute values appear. `individuals` writes one class per attribute and one individual per value carrying its literal, which the event or object points at. `literals` writes the literal on the event or object itself. Both can be combined. `equal_bins` buckets numbers by frequency, `dist_bins` by distance from the typical value, in standard deviations after a power transform, so the outer bins hold the unusual values. A bin mode needs `individuals`, and at most one can be set. Each attribute is cut once, over every value it takes anywhere in the log. A column with too few distinct values to bin keeps its numbers. Under `OWL2DL` each bin class is also defined by its value range, so a reasoner places a value in its bin. |
| `inverse_representations` | none | `axiomatic` writes inverse property axioms (OWL2DL only), `implicit` materializes the inverse assertions. |
| `ocel_extensions` | none | `attribute_history` also emits the timestamped attribute changes of each object. With or without it, an object carries the value of each attribute with the greatest timestamp. |
| `bins`, `bin_labels` | `5`, `very_low` … `very_high` | Bin count and labels for the two binning modes. |
| `max_ef_pairs` | `50000000` | Ceiling on materialized eventually-follows pairs. Zero for no limit. |
| `size_limit_mb` | `0` | Skip inputs larger than this many MB. Zero for no limit. |
| `derive_annotations` | `false` | Emit `rdfs:label` and `rdfs:comment` derived from the original names in the log. |
| `annotation_document` | none | JSON file mapping entity names to `{label, comment}`. |
| `declare_file` | none | File written by `semlog mine-declare`. Required for `declare`. |
| `materialize` | `false` | Run a reasoner over the result and save what it entails. Needs a JVM. |
| `reasoner_backend` | `HermiT` | Reasoner `materialize` runs: `HermiT`, `Pellet`, `ELK`. |
| `streaming` | `true` | Write OWL functional syntax to disk as `<name>.ofn`. Off builds the ontology in memory and writes `<name>.owl` in RDF/XML, which rdflib reads. |
| `save_backmap` | `false` | Also write the map from cleaned names to source names as JSON. |
| `log_dir` | none | Directory for a log file per input. |

`semlog build --help` documents every flag.

## From Python

```python
from semanticlog import (
    BuildInputs,
    EventOrderRepresentation,
    OntoBuildSettings,
    build_log,
    choose_case_concepts,
    load_and_clean,
)

data = load_and_clean("tests/fixtures/pipeline.jsonocel")
case_concepts, _ = choose_case_concepts(data, None)
settings = OntoBuildSettings(order_representations=[EventOrderRepresentation.DF])
result = build_log(data, settings, BuildInputs(case_concepts), "examples/out/pipeline.ofn")
```

`load_and_clean` reads the log and renames every name that cannot be part of an IRI. `choose_case_concepts` follows those renames for the types you name, or picks the most common types when given `None`. `OntoBuildSettings` holds the choices that do not depend on the log. `BuildInputs` holds the case concepts, the mined DECLARE constraints and any annotations. `build_log` writes the file once the build is complete. Pass `streaming=False` for RDF/XML, which also leaves the ontology on `result.onto`.

## Limits

- HermiT is the default reasoner and rejects a timestamp with microseconds. Set `reasoner_backend = "Pellet"` for such a log.
- ELK answers class queries only. It does not entail disjointness, inverses, data properties or property assertions.
- `--no-streaming` builds in memory and is about a hundred times slower on a large log.

## DECLARE support

`semlog mine-declare` needs the `[declare]` extra and the [Lydia](https://github.com/whitemech/lydia) LTLf/LDLf solver. Every other command works without them. Install the extra with:

```bash
uv sync --extra declare
```

Lydia is not on PyPI. You build it from source against CUDD 3.0.0, MONA 1.4 and Syft, which need `cmake`, `build-essential`, `bison`, `flex`, `libboost-dev` and `graphviz-dev`. [Declare4Py](https://github.com/ivanDonadello/Declare4Py) documents the setup.

## Development

```bash
make install
make check
```

`make install` installs the dev dependencies and the git hooks. `make check` runs lint, type check and the whole suite. `make help` lists every target. The reasoner tests need a JVM and skip themselves without one.

## License

MIT. See [LICENSE](LICENSE).
