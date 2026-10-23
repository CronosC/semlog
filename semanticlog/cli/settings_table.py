"""The table of build settings: key, default, reader, flag help and destination.

`BUILD_SETTINGS` declares every setting a config file or a flag can name.
`BUILD_DEFAULTS` is read from it, and the key groups below say which commands
take which settings.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from semanticlog.core.enums import (
    REASONER_BACKENDS,
    AttributeRepresentation,
    BuildEnum,
    EventOrderRepresentation,
    InverseRepresentation,
    OCELExtension,
    OWLProfile,
    resolve_reasoner_backend,
)
from semanticlog.core.errors import ConfigError
from semanticlog.core.vocabulary import TBOX_IRI, default_iri
from semanticlog.data.binning import DEFAULT_BIN_LABELS, DEFAULT_BINS
from semanticlog.data.loading import LOG_SUFFIXES
from semanticlog.onto.settings import DEFAULT_MAX_EF_PAIRS, OntoBuildSettings

_SETTINGS_DEFAULTS: dict[str, Any] = OntoBuildSettings().to_dict()


def settings_default(key: str) -> Any:
    """Return the default of one `OntoBuildSettings` field, as a config writes it.

    Args:
        key: Field name, which is also its TOML key.

    Returns:
        The value, with an enum as its spelling and a tuple as a list.

    Raises:
        KeyError: If the settings hold no such field.
    """
    return _SETTINGS_DEFAULTS[key]


def enum_choices(enum_cls: type[BuildEnum]) -> str:
    """Return the spellings of an enum for help text, one per member."""
    seen: set[str] = set()
    kept: list[str] = []
    for spelling in enum_cls.spellings():
        if spelling.casefold() not in seen:
            seen.add(spelling.casefold())
            kept.append(spelling)
    return " | ".join(kept)


def _as_list(value: Any) -> list[Any]:
    """Read a setting as a list, accepting a single value or nothing."""
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Iterable):
        return [value]
    return list(value)


def _as_bool(value: Any, key: str) -> bool:
    """Read a setting as a boolean, naming it if it is not one."""
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{key}: expected true or false, got {value!r}.")


def _as_int(value: Any, key: str) -> int:
    """Read a setting as a whole number, naming it if it is not one."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key}: expected a whole number, got {value!r}.")
    return value


_SUPPORT_RANGE = "a share of the traces above 0 and up to 1"


def _as_support(value: Any, key: str) -> float:
    """Read a support threshold, which is a share of the traces."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{key}: expected a number, got {value!r}.")
    if not 0 < value <= 1:
        raise ConfigError(f"{key}: expected {_SUPPORT_RANGE}, got {value}.")
    return float(value)


def _support_flag(value: str) -> float:
    """Read a support threshold flag. Raises `ArgumentTypeError`."""
    try:
        share = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if not 0 < share <= 1:
        raise argparse.ArgumentTypeError(f"expected {_SUPPORT_RANGE}, got {share}")
    return share


def _as_str(value: Any, key: str) -> str:
    """Read a setting as text, naming it if it is not."""
    if not isinstance(value, str):
        raise ConfigError(f"{key}: expected a string, got {value!r}.")
    return value


def _as_path(value: Any, key: str) -> Path:
    """Read a setting as a filesystem path."""
    return Path(_as_str(value, key))


def _as_reasoner_backend(value: Any, key: str) -> str:
    """Read a reasoner name in any case."""
    return resolve_reasoner_backend(_as_str(value, key), key=key)


Reader = Callable[[Any, str], Any]

type Target = Literal["settings", "run"]


def _optional(read: Reader) -> Reader:
    """Return a reader that passes None through."""

    def reader(value: Any, key: str) -> Any:
        return None if value is None else read(value, key)

    return reader


def _list_of(read: Reader) -> Reader:
    """Apply a reader to every element, accepting a single value or nothing."""

    def reader(value: Any, key: str) -> list:
        return [read(item, key) for item in _as_list(value)]

    return reader


def _enum_of[E: BuildEnum](enum_cls: type[E]) -> Reader:
    """Read a written setting as a member of one enum."""

    def reader(value: Any, key: str) -> E:
        return enum_cls.parse(_as_str(value, key), key=key)

    return reader


def _read_inputs(value: Any, key: str) -> tuple[Path, ...]:
    """Read the input setting as the files it names, directories expanded."""
    return expand_inputs(_list_of(_as_str)(value, key))


def _read_case_concepts(value: Any, key: str) -> tuple[str, ...] | None:
    """Read the case concepts, keeping None apart from an empty selection."""
    return None if value is None else tuple(_list_of(_as_str)(value, key))


def expand_inputs(tokens: Sequence[str]) -> tuple[Path, ...]:
    """Return the files the input setting names.

    A directory contributes every event log in it. Duplicates are removed by
    resolved path.

    Raises:
        ConfigError: If no input is given, a path is missing, is not an event
            log, or is a directory holding none, or two inputs share a stem.
    """
    if not tokens:
        raise ConfigError(
            "No input given. Pass a file or directory, or set 'input' in the config."
        )

    inputs: list[Path] = []
    for token in tokens:
        path = Path(_as_str(token, "input"))
        if path.is_dir():
            found = sorted(
                child
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in LOG_SUFFIXES
            )
            if not found:
                raise ConfigError(
                    f"{path} contains no event logs "
                    f"({', '.join(sorted(LOG_SUFFIXES))})."
                )
            inputs.extend(found)
        elif path.is_file():
            if path.suffix.lower() not in LOG_SUFFIXES:
                raise ConfigError(
                    f"{path} is not an event log semlog reads "
                    f"({', '.join(sorted(LOG_SUFFIXES))})."
                )
            inputs.append(path)
        else:
            raise ConfigError(f"Input not found: {path}. Check the path.")

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in inputs:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    _check_distinct_stems(unique)
    return tuple(unique)


def _check_distinct_stems(inputs: Sequence[Path]) -> None:
    """Raise if two inputs share a stem.

    Raises:
        ConfigError: If two inputs share a stem, naming them.
    """
    by_stem: dict[str, list[Path]] = {}
    for path in inputs:
        by_stem.setdefault(path.stem, []).append(path)
    clashing = {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}
    if not clashing:
        return
    lines = [
        f"  {stem}: " + ", ".join(str(p) for p in paths)
        for stem, paths in sorted(clashing.items())
    ]
    raise ConfigError(
        "These inputs share a stem:\n"
        + "\n".join(lines)
        + "\n\nBuild them one at a time with --name, or into separate --output "
        "directories."
    )


@dataclass(frozen=True)
class Setting:
    """One build setting: its key, default, reader, flag help and destination.

    Attributes:
        key: TOML key, and the argparse destination the flag writes to.
        default: Value a run uses when neither a config nor a flag names it.
        read: Turns a written value into the type the setting has.
        target: `settings` for a field of `OntoBuildSettings`, `run` for a
            field of `RunConfig`.
        help: Help text for the flag.
        argparse: Further arguments for `add_argument`. None for a setting
            whose flag the parser declares itself.
        field: Field this lands on, when it differs from the key.
    """

    key: str
    default: Any
    read: Reader
    target: Target
    help: str = ""
    argparse: Mapping[str, Any] | None = None
    field: str | None = None

    @property
    def flag(self) -> str:
        """The command line spelling of this setting."""
        return "--" + self.key.replace("_", "-")


BUILD_SETTINGS: tuple[Setting, ...] = (
    Setting(
        "input",
        None,
        _read_inputs,
        "run",
        field="inputs",
        help=(
            "Event logs or directories to read. A directory contributes every "
            "log in it. Omit to use the 'input' key of --config."
        ),
    ),
    Setting(
        "output",
        ".",
        _as_path,
        "run",
        field="output_dir",
        help="Directory for the ontology (default: the current directory).",
        argparse={"metavar": "DIR"},
    ),
    Setting(
        "name",
        None,
        _optional(_as_str),
        "run",
        help=(
            "Shared stem for every artifact of this run (default: the --config "
            "file name for a single input, otherwise the input name)."
        ),
        argparse={"metavar": "STEM"},
    ),
    Setting(
        "iri",
        None,
        _optional(_as_str),
        "run",
        help=(
            f"Base IRI of every name from the log (default: {default_iri('STEM')}, "
            "with STEM the artifact stem). A trailing / is added when missing."
        ),
        argparse={"metavar": "IRI"},
    ),
    Setting(
        "tbox_iri",
        settings_default("tbox_iri"),
        _as_str,
        "settings",
        help=(
            f"Namespace of the schema classes and properties (default: {TBOX_IRI}). "
            "Ends in # or /."
        ),
        argparse={"metavar": "IRI"},
    ),
    Setting(
        "profile",
        settings_default("profile"),
        _enum_of(OWLProfile),
        "settings",
        help=(
            f"OWL profile: {enum_choices(OWLProfile)} "
            f"(default: {settings_default('profile')})."
        ),
        argparse={"metavar": "PROFILE"},
    ),
    Setting(
        "order_representations",
        settings_default("order_representations"),
        _list_of(_enum_of(EventOrderRepresentation)),
        "settings",
        help=(
            "Event order representations: "
            f"{enum_choices(EventOrderRepresentation)} (default: none). "
            "Pass no values for none."
        ),
        argparse={"nargs": "*", "metavar": "MODE"},
    ),
    Setting(
        "case_concepts",
        None,
        _read_case_concepts,
        "run",
        help=(
            "Object types to use as cases, spelled as the log spells them "
            "(default: the most common types). Pass --case-concepts with no "
            "values for none. Run 'semlog inspect' to see the names."
        ),
        argparse={"nargs": "*", "metavar": "TYPE"},
    ),
    Setting(
        "auto_case_concepts",
        3,
        _as_int,
        "run",
        help="How many object types to auto-select (default: 3).",
        argparse={"type": int, "metavar": "N"},
    ),
    Setting(
        "case_col",
        None,
        _optional(_as_str),
        "run",
        help=(
            "Flat log only: column holding the case id (default: case:concept:name)."
        ),
        argparse={"metavar": "COLUMN"},
    ),
    Setting(
        "act_col",
        None,
        _optional(_as_str),
        "run",
        help="Flat log only: column holding the activity (default: concept:name).",
        argparse={"metavar": "COLUMN"},
    ),
    Setting(
        "time_col",
        None,
        _optional(_as_str),
        "run",
        help="Flat log only: column holding the timestamp (default: time:timestamp).",
        argparse={"metavar": "COLUMN"},
    ),
    Setting(
        "object_cols",
        [],
        _list_of(_as_str),
        "run",
        help=(
            "Flat log only: further columns that each become an object type "
            "(default: none)."
        ),
        argparse={"nargs": "*", "metavar": "COLUMN"},
    ),
    Setting(
        "attribute_representations",
        settings_default("attribute_representations"),
        _list_of(_enum_of(AttributeRepresentation)),
        "settings",
        help=(
            "Attribute representations: "
            f"{enum_choices(AttributeRepresentation)} (default: individuals). "
            "individuals writes one individual per value carrying its literal, "
            "literals writes the literal on the event or object, a bin mode "
            "adds bin classes for numbers and needs individuals. At most one "
            "bin mode."
        ),
        argparse={"nargs": "+", "metavar": "MODE"},
    ),
    Setting(
        "bins",
        settings_default("bins"),
        _as_int,
        "settings",
        help=f"Bin count for the binning modes (default: {DEFAULT_BINS}).",
        argparse={"type": int, "metavar": "N"},
    ),
    Setting(
        "bin_labels",
        settings_default("bin_labels"),
        _list_of(_as_str),
        "settings",
        help=(
            "Bin labels, exactly --bins of them "
            f"(default: {' '.join(DEFAULT_BIN_LABELS)})."
        ),
        argparse={"nargs": "+", "metavar": "LABEL"},
    ),
    Setting(
        "inverse_representations",
        settings_default("inverse_representations"),
        _list_of(_enum_of(InverseRepresentation)),
        "settings",
        help=(
            "Inverse relation representations: "
            f"{enum_choices(InverseRepresentation)} (default: none). "
            "axiomatic adds InverseObjectProperties axioms, OWL2DL only. "
            "implicit writes the reversed assertions."
        ),
        argparse={"nargs": "*", "metavar": "MODE"},
    ),
    Setting(
        "ocel_extensions",
        settings_default("ocel_extensions"),
        _list_of(_enum_of(OCELExtension)),
        "settings",
        help=(f"OCEL 2.0 extensions: {enum_choices(OCELExtension)} (default: none)."),
        argparse={"nargs": "*", "metavar": "EXT"},
    ),
    Setting(
        "annotation_document",
        None,
        _optional(_as_str),
        "run",
        help=(
            "JSON file mapping entity names to {label, comment} objects "
            "(default: none)."
        ),
        argparse={"metavar": "JSON"},
    ),
    Setting(
        "shared_attributes",
        settings_default("shared_attributes"),
        _as_bool,
        "settings",
        help=(
            "Also emit one class and one property per attribute name above the "
            "per-type ones, so a query over has_<name> reaches every type's "
            "value (default: off)."
        ),
        argparse={"action": argparse.BooleanOptionalAction},
    ),
    Setting(
        "derive_annotations",
        settings_default("derive_annotations"),
        _as_bool,
        "settings",
        help=(
            "Write rdfs:label and rdfs:comment on the classes and properties "
            "taken from the log, and rdfs:label on individuals (default: off)."
        ),
        argparse={"action": argparse.BooleanOptionalAction},
    ),
    Setting(
        "declare_file",
        None,
        _optional(_as_str),
        "run",
        help=(
            "DECLARE constraints file, as written by 'semlog mine-declare' "
            "(default: none). Required when --order-representations includes "
            "declare."
        ),
        argparse={"metavar": "FILE"},
    ),
    Setting(
        "constraints",
        None,
        _optional(_list_of(_as_str)),
        "run",
        help=(
            "Constraint templates to keep, e.g. Response Precedence Init End "
            "(default: all)."
        ),
        argparse={"nargs": "+", "metavar": "NAME"},
    ),
    Setting(
        "constraint_support",
        0.5,
        _as_support,
        "run",
        help="Minimum per-constraint support, above 0 and up to 1 (default: 0.5).",
        argparse={"type": _support_flag, "metavar": "T"},
    ),
    Setting(
        "itemset_support",
        0.5,
        _as_support,
        "run",
        help="Minimum itemset support, above 0 and up to 1 (default: 0.5).",
        argparse={"type": _support_flag, "metavar": "T"},
    ),
    Setting(
        "max_ef_pairs",
        settings_default("max_ef_pairs"),
        _as_int,
        "settings",
        help=(
            "Maximum number of eventually-follows pairs "
            f"(default: {DEFAULT_MAX_EF_PAIRS:,}). Zero for no limit."
        ),
        argparse={"type": int, "metavar": "N"},
    ),
    Setting(
        "materialize",
        False,
        _as_bool,
        "run",
        help="Run the reasoner over each ontology after writing it (default: off).",
        argparse={"action": argparse.BooleanOptionalAction},
    ),
    Setting(
        "reasoner_backend",
        "HermiT",
        _as_reasoner_backend,
        "run",
        help=f"Reasoner backend: {' | '.join(REASONER_BACKENDS)} (default: HermiT).",
        argparse={"metavar": "NAME"},
    ),
    Setting(
        "streaming",
        True,
        _as_bool,
        "run",
        help=(
            "Write functional syntax (.ofn) to disk as the build runs "
            "(default: on). --no-streaming builds in memory and writes RDF/XML "
            "(.owl)."
        ),
        argparse={"action": argparse.BooleanOptionalAction},
    ),
    Setting(
        "save_backmap",
        False,
        _as_bool,
        "run",
        help="Write the map of cleaned names to source names as JSON (default: off).",
        argparse={"action": argparse.BooleanOptionalAction},
    ),
    Setting(
        "size_limit_mb",
        0,
        _as_int,
        "run",
        help="Skip inputs larger than this many MB (default: 0, no limit).",
        argparse={"type": int, "metavar": "MB"},
    ),
    Setting(
        "log_dir",
        None,
        _optional(_as_path),
        "run",
        help="Directory for per-input log files (default: no file logging).",
        argparse={"metavar": "DIR"},
    ),
)

BUILD_DEFAULTS: dict[str, Any] = {s.key: s.default for s in BUILD_SETTINGS}

_BY_KEY: dict[str, Setting] = {s.key: s for s in BUILD_SETTINGS}

LOG_INPUT_KEYS: tuple[str, ...] = (
    "input",
    "case_col",
    "act_col",
    "time_col",
    "object_cols",
    "ocel_extensions",
    "case_concepts",
    "auto_case_concepts",
)

COLUMN_KEYS: tuple[str, ...] = ("case_col", "act_col", "time_col", "object_cols")

DECLARE_MINING_KEYS: tuple[str, ...] = (
    "constraints",
    "constraint_support",
    "itemset_support",
)


def setting(key: str) -> Setting:
    """Return the setting one key names."""
    return _BY_KEY[key]


def settings_targeting(target: Target) -> tuple[Setting, ...]:
    """Return the settings that land on one object."""
    return tuple(s for s in BUILD_SETTINGS if s.target == target)


def read_setting(merged: Mapping[str, Any], key: str) -> Any:
    """Read one setting out of a merged configuration, as the type it has."""
    return _BY_KEY[key].read(merged[key], key)
