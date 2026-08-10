# semlog

`semlog` converts Object-Centric Event Data (OCED/OCEL) from process mining into OWL 2 ontologies. It takes event logs in flat CSV/XES or OCEL 2.0 format, normalises them into an in-memory data model, and produces a `.owl` file that can be loaded into a reasoner, queried via SPARQL, or used as a knowledge base for downstream systems.

---

## Installation

```bash
# With uv (recommended)
uv sync                     # core + dev tools
uv sync --extra declare     # add DECLARE constraint support

# With pip
pip install -e .
pip install -e ".[dev]"
pip install -e ".[declare]"
```

Requires Python >= 3.12.

### DECLARE support

The `[declare]` extra installs `declare4py` but also requires the **Lydia** LTLf/LDLf solver, which cannot be installed via pip. Use the provided install script:

```bash
sudo bash scripts/install_lydia.sh
```

The script builds and installs from source: CUDD 3.0.0, MONA 1.4, Syft 0.1.1, and Lydia. Requires `cmake`, `build-essential`, `bison`, `flex`, `libboost-dev`, and `graphviz-dev` (installed automatically on Debian/Ubuntu). The rest of the library works without Lydia.

---

## Quick Start

The CLI is the entry point. From a fresh clone, this works with no extra data:

```bash
semlog inspect tests/fixtures/pipeline.jsonocel   # what is in this log?
semlog build --config examples/pipeline.toml      # build it
semlog reason examples/out/pipeline.owl --check   # is the result consistent?
```

Every build writes two files side by side:

| File | Contents |
|---|---|
| `<name>.owl` | The ontology — the product |
| `<name>.manifest.json` | How it was produced: input path and SHA-256, resolved settings, where the case concepts came from, semlog version, per-stage timings, entity counts |

The manifest is what makes a result traceable. An `.owl` on its own says nothing
about the configuration behind it, and two configurations of the same log
produce different ontologies.

See [CLI](#cli--semlog) for the full command reference.

---

## Library Use

```python
from semanticlog.data.loading.pm4pyloader import PM4PYLoader
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.settings import (
    DEFAULT_IRI, OntoBuildSettings, OwlProfile, EventOrderRepresentation
)

oced = PM4PYLoader("log.json", {}).get_oced()
oced = clean_oced_data(oced)

settings = OntoBuildSettings()
settings.profile = OwlProfile.OWL2DL
settings.case_concepts = ["order"]   # must be set for event order
settings.order_representation = [EventOrderRepresentation.DF]

semlog = OCEDOntoBuilder(DEFAULT_IRI, settings).build_ontology(oced)
semlog.write_ontology("output.owl")
```

The `SemanticLog` object returned by `build_ontology` carries the built ontology together with the settings used to produce it. Use `semlog.write_ontology(path)` to save the OWL file, or `semlog.write(path)` / `SemanticLog.read(path)` for a full pickle round-trip (`.semlog` files).

---

## Architecture

Core pipeline: load event data → clean/normalise → configure build settings → `OCEDOntoBuilder.build_ontology()` → `SemanticLog` → save `.owl`.

| Module | Role |
|---|---|
| `data/oced.py` | `OCEDStore` — central in-memory data model (`EventTable` / `ObjectTable` / `RelationTable`) |
| `data/dtypes.py` | `infer_dtypes()` — dtype coercion applied on ingest |
| `data/loading/classicalloader.py` | Load flat CSV/XES event logs |
| `data/loading/pm4pyloader.py` | Load OCEL 2.0 XML / JSON / SQLite via pm4py |
| `data/cleaning.py` | String cleaning, deduplication, collision-fixing |
| `data/filtering.py` | Filter by time window / variant frequency (operates on flat DataFrames) |
| `data/binning.py` | Variant weighting, numeric binning helpers |
| `onto/settings.py` | `OntoBuildSettings` (incl. `validate()`), all enums |
| `onto/semlog.py` | `SemanticLog` — pipeline output; write/read OWL + pickle |
| `onto/upper.py` | Upper ontology (TBox schema) builder; called internally by `OCEDOntoBuilder` |
| `onto/instantiate.py` | Full pipeline entry point: `OCEDOntoBuilder.build_ontology()` |
| `onto/sink.py` | `AxiomSink` — the one place a name becomes an axiom; `InMemorySink` (owlapy) by default, `StreamingSink` for large logs |
| `onto/emit.py` | `Emitter` — base for the build steps that write axioms |
| `onto/abox.py` | ABox emitters: events, objects, time, event order, relations |
| `onto/audit.py` | Pre-build rejection of malformed or ambiguous IRI local names |
| `onto/provenance.py` | Version and build configuration recorded into the ontology |
| `data/persist.py` | `OCEDStore.write`/`read` — Parquet tables plus a JSON manifest |
| `onto/builder.py` | Builder facade: holds the sink, settings and annotation writer |
| `onto/vocabulary.py` | Every IRI local name a build emits; source of the reserved-name set |
| `onto/annotations.py` | `AnnotationWriter` — all `rdfs:label` / `rdfs:comment` text |
| `onto/attributes.py` | Attribute classes, properties and value individuals |
| `onto/history.py` | OCEL 2.0 object attribute change history |
| `onto/transform.py` | Manchester Syntax → OWL expression parsing |
| `onto/reason.py` | Reasoner integration |
| `enrich/declare.py` | `DeclareResult` + `DeclareEnricher` — DECLARE constraint mining (requires Lydia) |
| `cli/__init__.py` | Argument parsing and subcommand dispatch — the only module that knows about argparse |
| `cli/config.py` | `RunConfig` — TOML and flags resolved into one checked value |
| `cli/run.py` | The pipeline behind each subcommand, driven by values rather than flags |
| `cli/manifest.py` | `RunManifest` — the record written beside every ontology |
| `cli/logs.py` | Console and per-input file logging |

---

## Data Loading

**`PM4PYLoader`** is the preferred loader. It reads OCEL 2.0 XML, JSON, or SQLite via pm4py; format is inferred from file extension.

```python
from semanticlog.data.loading.pm4pyloader import PM4PYLoader
from semanticlog.onto.settings import OcelExtension

oced = PM4PYLoader("log.sqlite", {}, ocel_extensions=[OcelExtension.ATTRIBUTE_HISTORY]).get_oced()
```

Always call `clean_oced_data(oced)` before building the ontology — it sanitises all names to `[A-Za-z0-9_]`, resolves collisions with reserved OWL property names, and deduplicates IDs. Skipping it produces malformed IRIs.

**`ClassicalLoader`** loads flat CSV/XES logs. It prefixes event IDs with `e` and case IDs with `t`.

The `ATTRIBUTE_HISTORY` extension loads per-object attribute change history from pm4py's `log.object_changes`. Without it, only the latest attribute value per object is loaded.

---

## Build Configuration

`OntoBuildSettings` controls what the ontology contains:

| Field | Type | Default | Description |
|---|---|---|---|
| `profile` | `OwlProfile` | `OWL2DL` | `OWL2DL` (full) or `OWL2EL` (lightweight, no TBox inverse/transitive axioms) |
| `order_representation` | `list[EventOrderRepresentation]` | `[]` | Event ordering strategies; combinable |
| `case_concepts` | `list[str]` | `[]` | Object types used as cases — required for any event order |
| `inverse_representation` | `list[InverseRepresentation]` | `[]` | `AXIOMATIC`: TBox axiom (OWL2DL only); `IMPLICIT`: ABox materialisation (both profiles). Empty means no inverses |
| `ocel_extensions` | `list[OcelExtension]` | `[]` | Optional OCEL 2.0 features (e.g. `ATTRIBUTE_HISTORY`) |
| `data_representation` | `list[DataRepresentation]` | `[VALUES]` | How numeric attributes are represented |
| `bins` | `int` | `5` | Bin count for `EQUAL_BINS` / `DIST_BINS` |
| `bin_labels` | `list[str]` | `["very_low", …, "very_high"]` | Labels for each bin; must have exactly `bins` elements |
| `annotation_document` | `str \| None` | `None` | Path to JSON mapping entity names to `{label, comment}` dicts |
| `annotation_data` | `dict \| None` | `None` | Annotation JSON; read from `annotation_document` on first access, or assign directly |
| `declare_file` | `str \| None` | `None` | Path to a DECLARE results CSV; read on first access to `declare_result` |
| `declare_result` | `DeclareResult \| None` | `None` | Loaded DECLARE constraints; required when `DECLARE` is in `order_representation` |
| `derive_annotations` | `bool` | `False` | Emit `rdfs:label` on event, Time, object, and attribute individuals |

`DataRepresentation` modes:
- `VALUES` — one named individual per distinct value; literal stored via `has_data_value`
- `EQUAL_BINS` — equal-frequency binning (`pd.qcut`); bin labels become OWL subclasses of the attribute class
- `DIST_BINS` — distribution-fitted binning (normal, log-normal, exponential, gamma via AIC); falls back to `EQUAL_BINS` when no distribution fits

`VALUES` and one binning mode can coexist in the same list. `EQUAL_BINS` and `DIST_BINS` are mutually exclusive. Columns with ≤ max(`bins`, 10) unique values are converted to strings directly without binning.

---

## Event Order Representations

Event order is only instantiated when `settings.case_concepts` is non-empty.

- **`DF`** — directly-follows: adjacent event pairs within each trace asserted via `df_{case}` sub-property
- **`EF`** — eventually-follows via TBox transitivity axioms; OWL2DL only (a reasoner derives all pairs); requires `DF` to also be included
- **`EF_explicit`** — same as EF but all pairs are materialised as ABox assertions; works in both profiles; increases ontology size for long traces
- **`DECLARE`** — DECLARE constraint membership: each trace is typed as the `DeclareConstraint` subclass for each constraint it satisfies; requires `settings.declare_result`

---

## DECLARE Constraint Mining

Requires the `[declare]` extra and Lydia (see [Installation](#installation)). Mining is a
separate step from building, so only this step needs Lydia and the mined constraints can
be reused across builds.

```bash
semlog mine-declare data/ocel/log.json \
    --output data/declare/log.csv \
    --constraints Response Precedence Coexistence Init End \
    --constraint-support 0.5 --itemset-support 0.5 \
    --case-concepts order

semlog build data/ocel/log.json --order declare --declare-file data/declare/log.csv
```

From Python:

```python
from semanticlog.enrich.declare import DeclareEnricher

result = DeclareEnricher(oced).get_constraints_via_discovery(
    case_concepts=["order"],
    constraint_set=["Response", "Precedence", "Coexistence", "Init", "End"],
    constraint_support_threshold=0.5,
    itemset_support_threshold=0.5,
)
result.write("constraints.csv")
settings.declare_file = "constraints.csv"   # read on first access to declare_result
```

`get_constraints_via_discovery` mines constraints meeting the support thresholds. `get_constraints_via_enumeration` checks a fixed constraint set exhaustively. The `DeclareResult` object can be serialised to CSV (`write`/`read`) or embedded in settings (`to_dict`/`from_dict`).

---

## CLI — `semlog`

Four subcommands. `semlog <command> --help` documents each in full.

| Command | Purpose |
|---|---|
| `semlog build` | Load, clean and build one or more ontologies, plus a manifest per result |
| `semlog inspect` | Summarise an OCEL file: activities, object types, relations, time range, suggested case concepts |
| `semlog mine-declare` | Mine DECLARE constraints into a CSV (needs Lydia) |
| `semlog reason` | Materialise or consistency-check an existing `.owl` |

### Run configs

Any build setting can live in a TOML file. `extends` pulls in shared settings, and a flag
on the command line always beats the file, so one config can be reused with a variation:

```toml
# runs/hiring.toml
extends = "_base.toml"

input = "data/ocel/Hiring.xml"
case_concepts = ["Candidate", "JobRequisition", "Recruiter"]
```

```bash
semlog build --config runs/hiring.toml
semlog build --config runs/hiring.toml --profile OWL2DL   # same run, DL instead
semlog build --config runs/hiring.toml --dry-run          # resolve and print, build nothing
```

The output is named after the config (`hiring.owl`, `hiring.manifest.json`), so several
configurations of one dataset coexist without overwriting each other. Relative paths in a
config resolve against the current directory; only `extends` resolves against the config
file. An unknown key is an error, with a suggestion.

`examples/pipeline.toml` is a complete config that runs from a fresh clone.

### `semlog build`

Input is positional: a file, a directory (every OCEL file in it), or omitted in favour of
the config's `input` key. A path that does not exist is an error rather than a silent skip.

| Flag | Config key | Default | Description |
|---|---|---|---|
| `PATH...` | `input` | from config | OCEL files or directories to build |
| `--config` | — | none | TOML config; flags override it |
| `--dry-run` | — | off | Print the resolved config and output paths, build nothing |
| `--output` | `output` | `data/onto` | Directory for the ontology and manifest |
| `--name` | `name` | config or input stem | Shared stem for every artifact |
| `--profile` | `profile` | `OWL2DL` | `OWL2DL` or `OWL2EL` |
| `--order` | `order` | `df` | `df ef ef_explicit declare`; pass with no values for none |
| `--case-concepts` | `case_concepts` | auto | Object types to use as cases |
| `--auto-case-concepts` | `auto_case_concepts` | `3` | How many to auto-select |
| `--data-representation` | `data_representation` | `values` | `values equal_bins dist_bins` |
| `--bins` | `bins` | `5` | Bin count for the binning modes |
| `--bin-labels` | `bin_labels` | `very_low … very_high` | Must have exactly `--bins` entries |
| `--inverse-representation` | `inverse_representation` | none | `axiomatic implicit` |
| `--ocel-extensions` | `ocel_extensions` | none | `attribute_history` |
| `--annotation-document` | `annotation_document` | none | JSON of entity name → label/comment |
| `--derive-annotations` | `derive_annotations` | off | Emit `rdfs:label` on individuals |
| `--declare-file` | `declare_file` | none | Constraints CSV; required with `--order declare` |
| `--materialize` | `materialize` | off | Run the reasoner after writing |
| `--reasoner-backend` | `reasoner_backend` | `HermiT` | `HermiT`, `Pellet` or `ELK` |
| `--save-backmap` | `save_backmap` | off | Also write the cleaned → original name map |
| `--save-store` | `save_store` | off | Also write the loaded `OCEDStore` pickle |
| `--size-limit` | `size_limit` | `0` | Skip inputs larger than N MB |
| `--log-dir` | `log_dir` | none | Directory for per-input log files |

Enum values are accepted in either spelling, case-insensitively: `--order DF` and
`--order df` are the same, as are `ATTRIBUTE_HISTORY` and `attribute_history`.

Exit codes: `0` success, `1` a build failed or an ontology was inconsistent, `2` the
configuration was unusable.

### Makefile targets

```bash
make example                        # build examples/pipeline.toml
make build CONFIG=runs/hiring.toml  # build one config
make build-all                      # build every runs/*.toml
```

---

## Tests

```bash
pytest tests/
```

---

## Dependencies

| Package | Version | Notes |
|---|---|---|
| `pm4py` | `>=2.7` | Process mining utilities, OCEL loading |
| `owlapy` | latest | OWL 2 ontology construction |
| `pandas` | `>=2.2` | All tabular data |
| `scipy` | latest | Distribution fitting for numeric binning |
| `declare4py` | optional | DECLARE constraint mining (`[declare]` extra; also needs Lydia) |
