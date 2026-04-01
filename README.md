# semlog

`semlog` converts Object-Centric Event Data (OCED/OCEL) from process mining into OWL 2 ontologies. It takes event logs in flat CSV/XES or OCEL 2.0 format, normalises them into an in-memory data model, and produces a `.owl` file that can be loaded into a reasoner, queried via SPARQL, or used as a knowledge base for downstream systems.

---

## Installation

```bash
pip install -e .
pip install -e ".[dev]"      # with dev tools
pip install -e ".[declare]"  # with DECLARE constraint support
```

Requires Python >= 3.12.

---

## Quick Start

```python
from semanticlog.data.classicalloader import ClassicalLoader
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.settings import OntoBuildSettings, OwlProfile, EventOrderRepresentation

cl = ClassicalLoader("data/log.csv", {
    "case_col": "case_id",
    "act_col": "activity",
    "time_col": "timestamp",
})
oced = cl.get_oced()
oced = clean_oced_data(oced)

settings = OntoBuildSettings()
settings.profile = OwlProfile.OWL2DL
settings.case_concepts = ["case_id"]   # must be set for event order
settings.order_representation = [EventOrderRepresentation.DF]

builder = OCEDOntoBuilder("https://example.org/myonto#", settings)
builder.build_ontology(oced)
builder.save_onto("output.owl")
```

The upper ontology (TBox schema) is built separately:

```python
from semanticlog.onto.upper import OCEDUpperOntoBuilder

upper = OCEDUpperOntoBuilder("https://example.org/upper#", settings)
upper.construct_upper_ontology("upper.owl")
```

---

## Architecture

The core pipeline: load event data → clean/normalise → configure build settings → instantiate OWL ontology → save `.owl` file.

| Module | Role |
|---|---|
| `data/oced.py` | `OCEDStore` — central in-memory data model |
| `data/classicalloader.py` | Load flat CSV/XES event logs |
| `data/ocelloader.py` | Load OCEL 2.0 SQLite directly |
| `data/pm4pyloader.py` | Load OCEL 2.0 XML / JSON / SQLite via pm4py |
| `data/cleaning.py` | String cleaning, deduplication, collision-fixing |
| `data/filtering.py` | Filter by activity / object type |
| `data/misc.py` | Variant weighting, numeric quantization |
| `onto/settings.py` | `OntoBuildSettings`, `OwlProfile`, `EventOrderRepresentation` |
| `onto/upper.py` | Upper ontology (schema) builder |
| `onto/instantiate.py` | ABox population pipeline |
| `onto/builder.py` | Low-level OWL axiom construction (owlapy) |
| `onto/transform.py` | String ↔ OWL expression parsing |
| `onto/reason.py` | Reasoner integration |
| `enrich/declare.py` | DECLARE constraint mining (requires Lydia backend) |

---

## Data Loading

### Loaders and config keys

**`ClassicalLoader`** — flat CSV or XES, one event per row.

| Key | Default | Description |
|---|---|---|
| `case_col` | `case:concept:name` | Column identifying the case/trace |
| `act_col` | `concept:name` | Column identifying the activity |
| `time_col` | `time:timestamp` | Column with event timestamps |
| `object_cols` | `[]` | Additional object-type columns |

Event attributes beyond the timestamp are not loaded by `ClassicalLoader` — only the case object and optional extra object columns are carried through.

**`OCELLoader`** — OCEL 2.0 SQLite, reads tables directly (no pm4py dependency).

| Key | Required | Description |
|---|---|---|
| `file_format` | yes | Must be `"sqlite"` (JSON and XML not yet implemented) |

**`PM4PYLoader`** — OCEL 2.0 XML, JSON, or SQLite via pm4py. Format is inferred from file extension if `file_format` is not given.

| Key | Default | Description |
|---|---|---|
| `file_format` | inferred | `"xml"`, `"json"`, or `"sqlite"` |

All loaders return an `OCEDStore`. Run `clean_oced_data(oced)` on the result before building the ontology.

### Known information losses

Both OCEL loaders make the following simplifications relative to the OCEL 2.0 standard:

- **Object attribute change history**: OCEL 2.0 stores time-varying object attributes as a sequence of change records, each with its own independent timestamp. Both `OCELLoader` and `PM4PYLoader` keep only the latest attribute value per object. The temporal evolution is discarded. (`OCELLoader` takes `idxmax` per object; `PM4PYLoader` inherits pm4py's flattened view.)
- **E2O relation timestamps** (`PM4PYLoader` only): OCEL 2.0 records when each event-object relation was observed, independently from the event timestamp. `PM4PYLoader` discards this. Representing it in OWL would require reifying each E2O relation as its own individual (n-ary relation pattern).

---

## Data Preprocessing

### Cleaning

`clean_oced_data(oced)` must be called before `build_ontology()`. It sanitises all names to `[A-Za-z0-9_]`, resolves collisions with reserved OWL property names, and deduplicates IDs. Skipping it will produce malformed IRIs or axiom collisions.

### Quantization

Numeric attributes produce a large number of distinct attribute individuals in the ontology (one per value). `quantize(oced)` replaces all numeric columns in events and objects with categorical bin labels before the ontology is built, collapsing the value space.

```python
from semanticlog.data.misc import quantize

oced = quantize(oced)   # call after clean_oced_data, before build_ontology
```

Binning strategy per column:
1. Fits candidate distributions (normal, log-normal, exponential, gamma) using AIC + KS test
2. If a known distribution fits, bins using normal-equivalent deviates
3. Falls back to equal-frequency quantile binning otherwise
4. Columns with low cardinality (≤ max(bins, 10) unique values) are converted to string labels directly

Default: 5 bins labelled `very_low`, `low`, `average`, `high`, `very_high`.

---

## Build Configuration

`OntoBuildSettings` controls what the ontology contains:

| Field | Type | Default | Description |
|---|---|---|---|
| `profile` | `OwlProfile` | `OWL2DL` | OWL 2 profile |
| `order_representation` | `list[EventOrderRepresentation]` | `[]` | Event ordering strategy/strategies |
| `case_concepts` | `list[str]` | `[]` | Object types used as cases — required for any event order |
| `declare_file` | `str \| None` | `None` | Path to DECLARE constraint CSV |
| `inverse_properties` | `bool` | `True` | Declare inverse object properties in TBox |
| `make_inverses_explicit` | `bool` | `True` | Materialise inverse assertions in ABox |

`OwlProfile`:
- `OWL2DL` — full OWL 2 DL: inverse axioms, reflexive/transitive closures, full reasoning
- `OWL2EL` — lightweight: no inverse axioms in TBox, no reflexive/transitive closures

`EventOrderRepresentation` values are combinable in a list: `DF`, `EF`, `EF_explicit`, `DECLARE`.

---

## Ontology Data Model

### Core class hierarchy

Always present, regardless of settings:

```
owl:Thing
├── Object          # one subclass per OCED object type
├── Event           # one subclass per activity name
├── Attribute
│   ├── ObjectAttribute   # one subclass per object attribute name
│   └── EventAttribute    # one subclass per event attribute name
└── Time
```

### Core properties

Object properties:

| Property | Domain | Range | Notes |
|---|---|---|---|
| `observed_at` | `Event` | `Time` | When the event occurred |
| `observes` | `Time` | `Event` | Inverse of `observed_at` |
| `event_object_relation` | `Event` | `Object` | Super-property for all E2O relations |
| `{qualifier}` | `Event` | `Object` | Sub-property per OCEL e2o qualifier name |
| `object_object_relation` | `Object` | `Object` | Super-property for all O2O relations |
| `{qualifier}_o2o` | `Object` | `Object` | Sub-property per OCEL o2o qualifier name |
| `has_attribute` | `Object or Event` | `Attribute` | Super-property for all attribute properties |
| `has_{attr}` | `Object or Event` | `{attr}` | Sub-property per attribute name |
| `attribute_of` | `Attribute` | `Object or Event` | Inverse of `has_attribute` |

Data properties:

| Property | Domain | Range | Notes |
|---|---|---|---|
| `has_data_value` | `Attribute or Time` | xsd literal | Typed literal for numeric/datetime attributes and Time individuals |

### ABox individual naming

| Individual type | Naming pattern | Example |
|---|---|---|
| Event | `{event_id}` | `e0`, `e1` |
| Object | `{object_id}` | `t42` |
| Attribute | `{attr_name}_{value}` | `price_100`, `status_open` |
| Time | `{YYYYMMDDTHHmmSS_microseconds}` | `20240101T090000_000000` |

Notes:
- `ClassicalLoader` prefixes event IDs with `e` and case IDs with `t`.
- `PM4PYLoader` prefixes purely numeric IDs with their type name (e.g., object `42` of type `Order` → `Order_42`) to avoid IRI collisions.
- Attribute individuals are **value-centric**: all entities with the same attribute value share one individual (`price_100` is shared by every object/event with `price=100`).
- `has_data_value` is only asserted for numeric and datetime attributes. String/boolean attribute individuals carry their value in their name only.

---

## Event Order Representations

Event order is only instantiated when `settings.case_concepts` is non-empty. All order representations start by adding:

```
Trace
└── Trace_{case_concept}    # one per entry in settings.case_concepts

part_of: Event → Trace
```

Each event individual is linked to its trace via `part_of`.

### Directly-follows (`EventOrderRepresentation.DF`)

Adds:
- `StartEvent ⊑ Event`, `EndEvent ⊑ Event`
- `df`: Event → Event (super-property)
- `df_{case}`: Event → Event (sub-property, one per case concept)

Adjacent event pairs within each trace are asserted via `df_{case}`. First/last events per trace are typed as `StartEvent` / `EndEvent`.

### Eventually-follows (`EventOrderRepresentation.EF`)

Adds `ef` and `ef_{case}` (DF must also be included).

In **OWL2DL**, `ef_{case}` is the reflexive, transitive closure of `df_{case}` — defined via TBox axioms, inferred by a reasoner; no ABox assertions added.

In **OWL2EL**, reflexive/transitive closure is not supported. Use `EF_explicit` instead.

### Eventually-follows explicit (`EventOrderRepresentation.EF_explicit`)

Same as EF but all `ef_{case}` pairs are materialised as ABox assertions (upper-triangular enumeration per trace). Works in both OWL2DL and OWL2EL. Significantly increases ontology size for long traces.

### DECLARE constraints (`EventOrderRepresentation.DECLARE`)

Requires `settings.declare_file` pointing to a CSV with columns:
`case_concept`, `trace_id`, `{constraint_name_1}`, `{constraint_name_2}`, ...

Each boolean column encodes whether a trace satisfies that DECLARE constraint. Constraints are instantiated separately per `case_concept` value in the file.

Adds:

```
Trace
└── DeclareConstraint
    ├── Response_A_B     # one subclass per constraint column
    ├── Precedence_C_D
    └── ...
```

A trace individual is additionally typed as each constraint class it satisfies:

```
trace_42  :  Trace_order_type    # from part_of
trace_42  :  Response_A_B        # this trace satisfies Response(A, B)
```

Satisfaction is encoded purely via class membership — no extra properties. Instances of `Response_A_B` are exactly the traces satisfying that constraint.

> Note: multi-case-concept DECLARE in a single CSV is not yet implemented.

---

## OWL Profile Differences

| Feature | OWL2DL | OWL2EL |
|---|---|---|
| Inverse property TBox axioms | Declared | Not declared |
| Inverse assertions in ABox (`make_inverses_explicit`) | Supported | Supported |
| `ef` as reflexive+transitive closure | Yes (TBox axioms) | Not supported — use `EF_explicit` |

---

## Reasoning and Materialization

After saving a `.owl` file, `Reasoner` can run an OWL reasoner over it to check consistency and materialise all inferred axioms back into the ontology.

```python
from semanticlog.onto.reason import Reasoner

r = Reasoner("output.owl", backend="HermiT")
r.check()           # check ontology consistency
r.materialize()     # infer and add all axioms, overwrite file in place

# or save to a new file
r.materialize(overwrite_old=False)
r.save_onto("output_materialized.owl")
```

Supported backends: `"HermiT"` (default), `"Pellet"`, `"ELK"`. ELK is the fastest and suited for OWL2EL ontologies. HermiT and Pellet support full OWL2DL but are slower on large ABoxes.

Inference types applied during materialization include class assertions, sub-class axioms, property assertions, inverse properties, and sub-property axioms. Property characteristic axioms (transitivity, reflexivity, etc.) are additionally included for OWL2DL ontologies.

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
| `scipy` | latest | Distribution fitting for quantization |
| `declare4py` | optional | DECLARE constraint mining (`pip install -e ".[declare]"`) |

`declare4py` additionally requires the **Lydia** LTLf/LDLf solver to be installed separately — it cannot be installed via pip alone. The rest of the library works without it.
