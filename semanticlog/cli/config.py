"""Turning defaults, a TOML file and flags into one checked configuration.

Precedence runs defaults, then the config file, then the flags. Everything is
resolved and validated here, so a run either has a `RunConfig` that describes
a build that can work, or it has failed with a message about the setting that
does not.

Configs can extend one another, which is how several runs over one dataset
share their common settings.
"""

from __future__ import annotations

import argparse
import difflib
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from semanticlog.core.enums import (
    REASONER_BACKENDS,
    BuildEnum,
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OCELExtension,
    OWLProfile,
    resolve_reasoner_backend,
)
from semanticlog.data.binning import (
    DEFAULT_BIN_LABELS,
    DEFAULT_BINS,
)
from semanticlog.data.loading.pm4py_loader import OCEL_SUFFIXES
from semanticlog.onto.settings import (
    DEFAULT_MAX_EF_PAIRS,
    FILE_BACKED,
    OntoBuildSettings,
)


class ConfigError(ValueError):
    """A setting is missing, unknown, or of the wrong sort."""


def parse_enum[E: BuildEnum](enum_cls: type[E], token: str, *, key: str) -> E:
    """Read an enum member from a written setting, as a configuration problem.

    The reading itself is `BuildEnum.parse`. This only restates its refusal as
    a `ConfigError`, so that a bad setting reaches the CLI the same way every
    other bad setting does.

    Raises:
        ConfigError: If nothing matches, listing what would.
    """
    try:
        return enum_cls.parse(token, key=key)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


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


def _as_str(value: Any, key: str) -> str:
    """Read a setting as text, naming it if it is not."""
    if not isinstance(value, str):
        raise ConfigError(f"{key}: expected a string, got {value!r}.")
    return value


def _as_path(value: Any, key: str) -> Path:
    """Read a setting as a filesystem path."""
    return Path(_as_str(value, key))


def _as_reasoner_backend(value: Any, key: str) -> str:
    """Read a reasoner name in any case, and reject an unknown one."""
    try:
        return resolve_reasoner_backend(_as_str(value, key), key=key)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


Reader = Callable[[Any, str], Any]

# Where a resolved setting lands: a field of `OntoBuildSettings`, a field of
# `RunConfig`, or neither, for one `resolve_run_config` assembles itself.
type Target = Literal["settings", "run", "manual"]


def _optional(read: Reader) -> Reader:
    """Let a reader accept None, which is how an unset setting arrives."""

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
        return parse_enum(enum_cls, _as_str(value, key), key=key)

    return reader


@dataclass(frozen=True)
class Setting:
    """One build setting, in the one place it is declared.

    Everything that knows about a setting reads it from here: the defaults a
    run starts from, the flag that overrides them, and the object the value
    ends up on.

    Attributes:
        key: TOML key, and the argparse destination the flag writes to.
        default: Value a run uses when neither a config nor a flag names it.
        read: Turns a written value into the type the setting has.
        target: `settings` for a field of `OntoBuildSettings`, `run` for a
            field of `RunConfig`, `manual` for one `resolve_run_config`
            assembles itself.
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
    Setting("input", None, _list_of(_as_str), "manual"),
    Setting(
        "output",
        "datasets/onto",
        _as_path,
        "run",
        field="output_dir",
        help="Directory for the ontology (default: datasets/onto).",
        argparse={"metavar": "DIR"},
    ),
    Setting(
        "name",
        None,
        _optional(_as_str),
        "manual",
        help=(
            "Shared stem for every artifact of this run. Defaults to the "
            "--config file name for a single input, otherwise to the input name."
        ),
        argparse={"metavar": "STEM"},
    ),
    Setting(
        "profile",
        OWLProfile.OWL2DL.value,
        _enum_of(OWLProfile),
        "settings",
        help=f"OWL profile: {' | '.join(OWLProfile.spellings())} (default: {OWLProfile.OWL2DL.value}).",
        argparse={"metavar": "PROFILE"},
    ),
    Setting(
        "order_representations",
        [EventOrderRepresentation.DF.value],
        _list_of(_enum_of(EventOrderRepresentation)),
        "settings",
        help=(
            "Event order representations: "
            f"{' | '.join(EventOrderRepresentation.spellings())}. "
            "Pass --order-representations with no values for none. Default: df."
        ),
        argparse={"nargs": "*", "metavar": "MODE"},
    ),
    Setting(
        "case_concepts",
        None,
        _optional(_list_of(_as_str)),
        "manual",
        help=(
            "Object types to use as cases. Omit to auto-select the most common "
            "types; pass --case-concepts with no values for none. Run "
            "'semlog inspect' to see the available names."
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
        "data_representations",
        [DataRepresentation.VALUES.value],
        _list_of(_enum_of(DataRepresentation)),
        "settings",
        help=(
            "How numeric attributes are represented: "
            f"{' | '.join(DataRepresentation.spellings())}. "
            "values keeps the raw number, equal_bins buckets by frequency, "
            "dist_bins fits a distribution first. equal_bins and dist_bins are "
            "mutually exclusive. Pass no values to omit numeric attributes. "
            "Default: values."
        ),
        argparse={"nargs": "*", "metavar": "MODE"},
    ),
    Setting(
        "bins",
        DEFAULT_BINS,
        _as_int,
        "settings",
        help=f"Bin count for the binning modes (default: {DEFAULT_BINS}).",
        argparse={"type": int, "metavar": "N"},
    ),
    Setting(
        "bin_labels",
        list(DEFAULT_BIN_LABELS),
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
        [],
        _list_of(_enum_of(InverseRepresentation)),
        "settings",
        help=(
            "Inverse relation strategies: "
            f"{' | '.join(InverseRepresentation.spellings())}. "
            "axiomatic adds InverseObjectProperty axioms (OWL2DL only), "
            "implicit materialises the inverse assertions. Default: none."
        ),
        argparse={"nargs": "*", "metavar": "MODE"},
    ),
    Setting(
        "ocel_extensions",
        [],
        _list_of(_enum_of(OCELExtension)),
        "settings",
        help=(
            f"OCEL 2.0 extensions: {' | '.join(OCELExtension.spellings())}. "
            "Default: none."
        ),
        argparse={"nargs": "*", "metavar": "EXT"},
    ),
    Setting(
        "annotation_document",
        None,
        _optional(_as_str),
        "settings",
        help="JSON file mapping entity names to {label, comment} dicts.",
        argparse={"metavar": "JSON"},
    ),
    Setting(
        "derive_annotations",
        False,
        _as_bool,
        "settings",
        help="Emit rdfs:label on individuals derived from the source data (default: off).",
        argparse={"action": argparse.BooleanOptionalAction},
    ),
    Setting(
        "declare_file",
        None,
        _optional(_as_str),
        "settings",
        help=(
            "DECLARE constraints CSV, as written by 'semlog mine-declare'. "
            "Required when --order-representations includes declare."
        ),
        argparse={"metavar": "CSV"},
    ),
    Setting(
        "max_ef_pairs",
        DEFAULT_MAX_EF_PAIRS,
        _as_int,
        "settings",
        help=(
            "Ceiling on materialized eventually-follows pairs, which grow with "
            f"the square of the trace length (default: {DEFAULT_MAX_EF_PAIRS:,}). "
            "Zero for no limit."
        ),
        argparse={"type": int, "metavar": "N"},
    ),
    Setting(
        "materialize",
        False,
        _as_bool,
        "run",
        help=(
            "Run the reasoner over each ontology after writing it (default: off). "
            "'semlog reason' does the same thing to an existing file."
        ),
        argparse={"action": argparse.BooleanOptionalAction},
    ),
    Setting(
        "reasoner_backend",
        "HermiT",
        _as_reasoner_backend,
        "run",
        help=f"Reasoner used by --materialize: {' | '.join(REASONER_BACKENDS)} (default: HermiT).",
        argparse={"metavar": "NAME"},
    ),
    Setting(
        "streaming",
        True,
        _as_bool,
        "run",
        help=(
            "Write OWL functional syntax straight to disk at near-constant "
            "memory (default: on). --no-streaming builds the ontology in "
            "memory and writes RDF/XML instead."
        ),
        argparse={"action": argparse.BooleanOptionalAction},
    ),
    Setting(
        "save_backmap",
        False,
        _as_bool,
        "run",
        help="Also write the cleaned-name to original-name map as JSON.",
        argparse={"action": "store_true"},
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


def settings_targeting(target: Target) -> tuple[Setting, ...]:
    """Return the settings that land on one object."""
    return tuple(s for s in BUILD_SETTINGS if s.target == target)


RENAMED_KEYS: dict[str, str] = {
    "order": "order_representations",
    "data_representation": "data_representations",
    "inverse_representation": "inverse_representations",
    "size_limit": "size_limit_mb",
}


def _check_keys(raw: dict[str, Any], source: Path) -> None:
    """Reject unknown settings, suggesting the nearest real one.

    A key that was renamed is matched by name rather than by similarity, since
    `order` and `order_representations` do not look alike enough for the
    fuzzy match to find it.
    """
    unknown = sorted(set(raw) - set(BUILD_DEFAULTS))
    if not unknown:
        return
    hints = []
    for key in unknown:
        renamed = RENAMED_KEYS.get(key)
        close = (
            [renamed]
            if renamed
            else difflib.get_close_matches(key, BUILD_DEFAULTS, n=1)
        )
        hints.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    raise ConfigError(
        f"{source}: unknown setting(s): {', '.join(hints)}.\n"
        f"Valid settings: {', '.join(sorted(BUILD_DEFAULTS))}."
    )


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Read a TOML config, following `extends` if it has one.

    Args:
        path: Config file to read.

    Returns:
        The settings it names, with its own entries winning over inherited
        ones.

    Raises:
        ConfigError: If the file is missing, is not valid TOML, names an
            unknown setting, or the chain of `extends` forms a cycle.
    """
    return _load_config_file(Path(path), [])


def _load_config_file(path: Path, chain: list[Path]) -> dict[str, Any]:
    """Read one config, tracking the chain already followed to detect a cycle."""
    resolved = path.resolve()
    if resolved in chain:
        trail = " -> ".join(str(p) for p in [*chain, resolved])
        raise ConfigError(f"config 'extends' forms a cycle: {trail}")
    if not resolved.is_file():
        raise ConfigError(f"config file not found: {path}")

    try:
        with resolved.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc

    extends = raw.pop("extends", None)
    _check_keys(raw, path)
    if extends is None:
        return raw
    if not isinstance(extends, str):
        raise ConfigError(f"{path}: 'extends' must be a path string.")
    base = _load_config_file(resolved.parent / extends, [*chain, resolved])
    return {**base, **raw}


@dataclass(frozen=True)
class RunConfig:
    """One resolved run: what to build, from what, and where to put it.

    Attributes:
        inputs: Log files to build, in order.
        output_dir: Directory every artifact goes into.
        name: Shared stem for the artifacts, or None to use each input name.
        settings: Build settings, apart from the case concepts, which are only
            known once the log is loaded.
        case_concepts: Object types to use as cases, or None to choose them
            from the log.
        auto_case_concepts: How many to choose when choosing automatically.
        materialize: Run a reasoner after writing.
        reasoner_backend: Reasoner to use for that.
        save_backmap: Also write the map of cleaned names to source names.
        streaming: Write the ontology straight to disk as functional syntax.
            Off builds it in memory and writes RDF/XML.
        size_limit_mb: Skip inputs larger than this. Zero for no limit.
        log_dir: Directory for per-input log files.
        config_file: Config this run came from, recorded as provenance.
        command: The command as typed, recorded as provenance.
        warnings: Anything worth saying before the run starts.
    """

    inputs: tuple[Path, ...]
    output_dir: Path
    name: str | None
    settings: OntoBuildSettings
    case_concepts: tuple[str, ...] | None
    auto_case_concepts: int = 3
    materialize: bool = False
    reasoner_backend: str = "HermiT"
    save_backmap: bool = False
    streaming: bool = True
    size_limit_mb: int = 0
    log_dir: Path | None = None
    config_file: Path | None = None
    command: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def output_stem(self, input_path: Path) -> str:
        """Return the shared name of the artifacts built from one input."""
        return self.name or input_path.stem

    def ontology_path(self, input_path: Path) -> Path:
        """Return where the ontology for one input goes."""
        return self.output_dir / f"{self.output_stem(input_path)}.owl"

    def backmap_path(self, input_path: Path) -> Path:
        """Return where the name backmap for one input goes."""
        return self.output_dir / f"{self.output_stem(input_path)}.backmap.json"

    def settings_for(self, case_concepts: Sequence[str]) -> OntoBuildSettings:
        """Return the settings with the case concepts this input resolved to."""
        return self.settings.replace(case_concepts=tuple(case_concepts))

    def render(self) -> str:
        """Render the run for reading, as printed by --dry-run.

        Returns:
            The settings that will be used and the files that would be written.
        """
        settings = self.settings
        case_concepts = (
            f"auto (the {self.auto_case_concepts} most common object types)"
            if self.case_concepts is None
            else (", ".join(self.case_concepts) or "none")
        )
        lines = [
            f"config file          {self.config_file or '(none, flags only)'}",
            f"inputs               {len(self.inputs)}",
            *(f"  - {path}" for path in self.inputs),
            f"output directory     {self.output_dir}",
            f"profile              {settings.profile.value}",
            f"order                {', '.join(r.value for r in settings.order_representations) or 'none'}",
            f"case concepts        {case_concepts}",
            f"data representation  {', '.join(r.value for r in settings.data_representations) or 'none'}",
            f"inverses             {', '.join(r.value for r in settings.inverse_representations) or 'none'}",
            f"ocel extensions      {', '.join(e.value for e in settings.ocel_extensions) or 'none'}",
            f"annotations          document={settings.annotation_document or 'none'} derived={settings.derive_annotations}",
            f"declare constraints  {settings.declare_file or 'none'}",
            f"materialize          {self.materialize}"
            + (f" ({self.reasoner_backend})" if self.materialize else ""),
        ]
        if settings.binning_mode is not None:
            lines.append(
                f"bins                 {settings.bins} ({', '.join(settings.bin_labels)})"
            )
        lines.append("")
        lines.append("would write:")
        for path in self.inputs:
            lines.append(f"  {self.ontology_path(path)}")
            if self.save_backmap:
                lines.append(f"  {self.backmap_path(path)}")
        return "\n".join(lines)


def _expand_inputs(tokens: Sequence[str]) -> tuple[Path, ...]:
    """Turn the input setting into the actual list of files to build.

    A directory contributes every OCEL file in it. A path that does not exist
    raises rather than yielding an empty result. Paths are compared resolved,
    so one file named two ways is built once.
    """
    if not tokens:
        raise ConfigError(
            "no input given. Pass a file or directory, or set 'input' in the config."
        )

    inputs: list[Path] = []
    for token in tokens:
        path = Path(_as_str(token, "input"))
        if path.is_dir():
            found = sorted(
                child
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in OCEL_SUFFIXES
            )
            if not found:
                raise ConfigError(
                    f"{path} contains no OCEL files "
                    f"({', '.join(sorted(OCEL_SUFFIXES))})."
                )
            inputs.extend(found)
        elif path.is_file():
            inputs.append(path)
        else:
            raise ConfigError(f"input not found: {path}")

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
    """Refuse inputs whose artifacts would land on one another.

    Outputs are named after the input, so two files of one name in different
    directories would write one ontology and report two builds.

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
        "these inputs would overwrite one another's output:\n"
        + "\n".join(lines)
        + "\n\nOutputs are named after the input. Build them one at a time with "
        "--name, or into separate --output directories."
    )


def _check_file_backed_settings(merged: dict[str, Any]) -> None:
    """Check the settings that name a file, before anything tries to read one.

    `OntoBuildSettings` opens each of these in `__post_init__`, so a missing
    file has to be caught here or it surfaces as an OS error instead of as a
    message about the setting.

    Raises:
        ConfigError: If a named file does not exist.
    """
    for setting in FILE_BACKED:
        value = merged.get(setting.path_field)
        if value is not None and not Path(_as_str(value, setting.path_field)).is_file():
            raise ConfigError(setting.missing.format(path=value))


def _settings_from(merged: dict[str, Any]) -> OntoBuildSettings:
    """Build the ontology settings from the merged configuration."""
    kwargs: dict[str, Any] = {}
    for setting in settings_targeting("settings"):
        kwargs[setting.key] = setting.read(merged[setting.key], setting.key)

    try:
        return OntoBuildSettings(**kwargs)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _binning_warnings(merged: dict[str, Any], settings: OntoBuildSettings) -> list[str]:
    """Warn about bin settings that were given but will not be used."""
    if settings.binning_mode is not None:
        return []
    if (
        merged["bins"] == BUILD_DEFAULTS["bins"]
        and merged["bin_labels"] == BUILD_DEFAULTS["bin_labels"]
    ):
        return []
    return [
        "bins/bin_labels have no effect without equal_bins or dist_bins in "
        f"data_representations (currently {[r.value for r in settings.data_representations]})."
    ]


def resolve_run_config(
    overrides: dict[str, Any],
    *,
    config_file: str | Path | None = None,
    command: str = "",
) -> RunConfig:
    """Resolve defaults, config file and flags into one checked run.

    The settings are validated without requiring the inputs that are only
    known once a log is loaded, since case concepts may still be chosen from
    the data. The builder validates again, with those included.

    Args:
        overrides: Settings given on the command line.
        config_file: TOML config to read first.
        command: The command as typed, recorded as provenance.

    Returns:
        The resolved run.

    Raises:
        ConfigError: If a setting is missing, unknown, of the wrong sort, or
            contradicts another one.
    """
    from_file = load_config_file(config_file) if config_file is not None else {}
    merged: dict[str, Any] = {**BUILD_DEFAULTS, **from_file, **overrides}

    inputs = _expand_inputs(_list_of(_as_str)(merged["input"], "input"))
    _check_file_backed_settings(merged)
    settings = _settings_from(merged)
    run: dict[str, Any] = {
        s.field or s.key: s.read(merged[s.key], s.key)
        for s in settings_targeting("run")
    }

    try:
        settings.validate(require_inputs=False)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    if settings.emits_declare and settings.declare_file is None:
        raise ConfigError(
            "order includes 'declare' but no declare_file is set. Mine the "
            "constraints first with 'semlog mine-declare', then point "
            "declare_file at the CSV it writes."
        )

    name = _optional(_as_str)(merged["name"], "name")
    if name is not None:
        if len(inputs) > 1:
            raise ConfigError(
                f"name={name!r} would make {len(inputs)} inputs share one output "
                "file. Drop it, or build one input at a time."
            )
    elif config_file is not None and len(inputs) == 1:
        name = Path(config_file).stem

    read_case_concepts = _optional(_list_of(_as_str))
    case_concepts = read_case_concepts(merged["case_concepts"], "case_concepts")
    if case_concepts is not None:
        case_concepts = tuple(case_concepts)
        if not case_concepts and settings.emits_traces:
            raise ConfigError(
                "case_concepts is empty but order needs traces "
                f"({', '.join(r.value for r in settings.order_representations)}). "
                "Name the object types to use as cases, or drop them from order."
            )

    return RunConfig(
        inputs=inputs,
        name=name,
        settings=settings,
        case_concepts=case_concepts,
        config_file=None if config_file is None else Path(config_file),
        command=command,
        warnings=tuple(_binning_warnings(merged, settings)),
        **run,
    )
