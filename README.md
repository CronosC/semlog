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
| `data/oced.py` | `OCEDStore` — central in-memory data model |
| `data/loading/classicalloader.py` | Load flat CSV/XES event logs |
| `data/loading/pm4pyloader.py` | Load OCEL 2.0 XML / JSON / SQLite via pm4py |
| `data/cleaning.py` | String cleaning, deduplication, collision-fixing |
| `data/filtering.py` | Filter by activity / object type (operates on flat DataFrames) |
| `data/binning.py` | Variant weighting, numeric binning helpers |
| `onto/settings.py` | `OntoBuildSettings`, all enums, `validate_settings` |
| `onto/semlog.py` | `SemanticLog` — pipeline output; write/read OWL + pickle |
| `onto/upper.py` | Upper ontology (TBox schema) builder; called internally by `OCEDOntoBuilder` |
| `onto/instantiate.py` | Full pipeline entry point: `OCEDOntoBuilder.build_ontology()` |
| `onto/builder.py` | Low-level OWL axiom construction (owlapy) |
| `onto/transform.py` | String ↔ OWL expression parsing |
| `onto/reason.py` | Reasoner integration |
| `enrich/declare.py` | `DeclareResult` + `DeclareEnricher` — DECLARE constraint mining (requires Lydia) |

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
| `inverse_representation` | `list[InverseRepresentation]` | `[AXIOMATIC, IMPLICIT]` | `AXIOMATIC`: TBox axiom (OWL2DL only); `IMPLICIT`: ABox materialisation (both profiles) |
| `ocel_extensions` | `list[OcelExtension]` | `[]` | Optional OCEL 2.0 features (e.g. `ATTRIBUTE_HISTORY`) |
| `data_representation` | `list[DataRepresentation]` | `[VALUES]` | How numeric attributes are represented |
| `bins` | `int` | `5` | Bin count for `EQUAL_BINS` / `DIST_BINS` |
| `bin_labels` | `list[str]` | `["very_low", …, "very_high"]` | Labels for each bin; must have exactly `bins` elements |
| `annotation_document` | `str \| None` | `None` | Path to JSON mapping entity names to `{label, comment}` dicts |
| `annotation_data` | `dict \| None` | `None` | Loaded annotation JSON; call `load_annotation_document()` to populate |
| `declare_file` | `str \| None` | `None` | Path to a DECLARE results CSV; call `load_declare_file()` to load |
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

Requires the `[declare]` extra and Lydia (see [Installation](#installation)).

```python
from semanticlog.enrich.declare import DeclareEnricher

enricher = DeclareEnricher(oced)
declare_result = enricher.get_constraints_via_discovery(
    case_concepts=["order"],
    constraint_set=["Response", "Precedence", "Coexistence", "Init", "End"],
    constraint_support_threshold=0.5,
    itemset_support_threshold=0.5,
)
declare_result.write("constraints.csv")

settings.declare_file = "constraints.csv"
settings.load_declare_file()            # populates settings.declare_result
```

`get_constraints_via_discovery` mines constraints meeting the support thresholds. `get_constraints_via_enumeration` checks a fixed constraint set exhaustively. The `DeclareResult` object can be serialised to CSV (`write`/`read`) or embedded in settings (`to_dict`/`from_dict`).

---

## CLI — `build_onto.py`

Batch-builds ontologies from a folder of OCEL files:

```bash
python scripts/build_onto.py --folder data/ocel --files log.json \
    --profile OWL2EL --order DF \
    --case-concepts order item \
    --data-representation values \
    --output data/onto
```

| Argument | Default | Description |
|---|---|---|
| `--folder` | required | Folder containing OCEL input files |
| `--files` | all files | Filenames within `--folder` to process |
| `--output` | `data/onto` | Output directory for `.owl` files |
| `--profile` | `OWL2DL` | `OWL2DL` or `OWL2EL` |
| `--order` | `DF` | Space-separated: `DF EF EF_explicit DECLARE` |
| `--case-concepts` | auto (top 3) | Object type names to use as case concepts |
| `--inverse-representation` | `AXIOMATIC IMPLICIT` | Space-separated; pass none to omit inverses |
| `--data-representation` | `values` | Space-separated: `values equal_bins dist_bins` |
| `--bins` | `5` | Bin count for `equal_bins` / `dist_bins` |
| `--bin-labels` | `very_low … very_high` | Bin label names (must match `--bins`) |
| `--ocel-extensions` | none | `ATTRIBUTE_HISTORY` |
| `--materialize` / `--no-materialize` | on | Run reasoner after build |
| `--annotation-document` | none | JSON file mapping entity names to label/comment |
| `--derive-annotations` | off | Emit `rdfs:label` on individuals |
| `--declare-constraints` | none | DECLARE constraint types to mine (requires `DECLARE` in `--order`) |
| `--declare-constraint-support` | `0.5` | Per-constraint support threshold |
| `--declare-itemset-support` | `0.5` | Itemset support threshold |
| `--log-dir` | none | Directory for per-file log files |
| `--size-limit` | 0 (no limit) | Skip files larger than N MB |

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
