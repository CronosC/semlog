"""Resolves defaults, a TOML file and flags into one checked configuration.

Start at `resolve_run_config` for `build` and `resolve_log_inputs` for the
other commands that read a log. Flags win over the config file, which wins
over the defaults.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semanticlog.cli.settings_table import (
    BUILD_DEFAULTS,
    BUILD_SETTINGS,
    COLUMN_KEYS,
    Setting,
    read_setting,
    setting,
    settings_targeting,
)
from semanticlog.cli.toml import load_config_file, toml_value
from semanticlog.core.enums import OCELExtension, OutputFormat, ReasonerBackend
from semanticlog.core.errors import ConfigError
from semanticlog.core.vocabulary import default_iri
from semanticlog.data.declare import DeclareResult
from semanticlog.data.loading import CLASSICAL_SUFFIXES
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings

__all__ = [
    "LogInputs",
    "RunConfig",
    "resolve_log_inputs",
    "resolve_run_config",
]

_ANNOTATION_KEYS = {"label", "comment"}


def _merged_settings(
    overrides: Mapping[str, Any], config_file: str | Path | None
) -> dict[str, Any]:
    """Apply the config file and then the flags on top of the defaults.

    Raises:
        ConfigError: If the config file cannot be read, or no input is given
            and a flag took an input path as one of its values.
    """
    from_file = load_config_file(config_file) if config_file is not None else {}
    merged = {**BUILD_DEFAULTS, **from_file, **overrides}
    if not merged["input"]:
        _check_swallowed_input(overrides)
    return merged


def _check_swallowed_input(overrides: Mapping[str, Any]) -> None:
    """Raise if a flag that takes several values ends in an existing file.

    Such a flag, given before the input, takes the input path as a value.

    Raises:
        ConfigError: Naming the path and the flag.
    """
    for key, value in overrides.items():
        declared = setting(key)
        takes_several = (declared.argparse or {}).get("nargs") in ("*", "+")
        if not takes_several or not value:
            continue
        last = value[-1]
        if isinstance(last, str) and Path(last).is_file():
            raise ConfigError(
                f"No input given, and {declared.flag} took {last} as one of its "
                f"values. Put {last} before {declared.flag}."
            )


def _loader_columns(named: Mapping[str, Any]) -> dict[str, Any]:
    """Return the column names a flat log is read with, leaving out unset ones."""
    columns = dict(named)
    columns["object_cols"] = columns["object_cols"] or None
    columns["object_attribute_cols"] = columns["object_attribute_cols"] or None
    return {key: value for key, value in columns.items() if value is not None}


@dataclass(frozen=True)
class LogInputs:
    """Which logs a command reads, and how to read them.

    `inspect` and `mine-declare` read their settings from this. `build` reads
    `RunConfig`.

    Attributes:
        inputs: Log files to read, in order.
        columns: Column names for a flat log. Unset ones are left out.
        ocel_extensions: OCEL 2.0 extensions to load.
        case_concepts: Object types to use as cases, or None to choose them
            from the log.
        auto_case_concepts: How many to choose when choosing automatically.
        declare_file: File of mined constraints, which `mine-declare` writes.
        constraints: Constraint templates to keep, or None to keep every one
            discovery proposes.
        constraint_support: Share of traces a constraint has to hold in.
        itemset_support: Share of traces an activity set has to occur in.
        warnings: Warnings to log before the command runs.
    """

    inputs: tuple[Path, ...]
    columns: Mapping[str, Any]
    ocel_extensions: tuple[OCELExtension, ...]
    case_concepts: tuple[str, ...] | None
    auto_case_concepts: int
    declare_file: str | None
    constraints: list[str] | None
    constraint_support: float
    itemset_support: float
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def one_input(self) -> Path:
        """Return the single input, for a command that writes one file.

        Raises:
            ConfigError: If the run resolved to more than one log.
        """
        if len(self.inputs) > 1:
            named = ", ".join(str(path) for path in self.inputs)
            raise ConfigError(
                f"This command reads one log, but the input resolved to "
                f"{len(self.inputs)}: {named}. Name one of them."
            )
        return self.inputs[0]


def resolve_log_inputs(
    overrides: dict[str, Any],
    *,
    config_file: str | Path | None = None,
) -> LogInputs:
    """Resolve defaults, config file and flags into the logs to read.

    Args:
        overrides: Settings given on the command line.
        config_file: TOML config to read first.

    Raises:
        ConfigError: If a setting is missing, unknown or of the wrong sort.
    """
    merged = _merged_settings(overrides, config_file)

    def read(key: str) -> Any:
        return read_setting(merged, key, overrides)

    inputs = read("input")
    return LogInputs(
        inputs=inputs,
        columns=_loader_columns({key: read(key) for key in COLUMN_KEYS}),
        ocel_extensions=tuple(read("ocel_extensions")),
        case_concepts=read("case_concepts"),
        auto_case_concepts=read("auto_case_concepts"),
        declare_file=read("declare_file"),
        constraints=read("constraints"),
        constraint_support=read("constraint_support"),
        itemset_support=read("itemset_support"),
        warnings=tuple(_column_warnings(merged, inputs)),
    )


@dataclass(frozen=True)
class RunConfig:
    """One resolved run.

    Every field before `warnings` is a `BUILD_SETTINGS` entry and
    `resolve_run_config` supplies all of them. Defaults are declared in the
    table only.

    Attributes:
        inputs: Log files to build, in order.
        output_dir: Directory every artifact goes into.
        name: Shared stem for the artifacts, or None to use each input name.
        settings: The `OntoBuildSettings` of the build. Per-log inputs come
            from `build_inputs`.
        case_concepts: Object types to use as cases, or None to choose them
            from the log.
        iri: Base IRI of the names from the log, or None for the default
            of each input.
        auto_case_concepts: How many to choose when choosing automatically.
        case_col: Column holding the case id of a flat log.
        act_col: Column holding the activity of a flat log.
        time_col: Column holding the timestamp of a flat log.
        object_cols: Further columns of a flat log that each become an object
            type.
        object_attribute_cols: Columns of a flat log that describe an object
            rather than an event.
        annotation_document: JSON file of entity name to label and comment.
        declare_file: File of mined DECLARE constraints.
        constraints: Constraint templates `mine-declare` keeps. Not used by a
            build.
        constraint_support: Share of traces a constraint has to hold in.
        itemset_support: Share of traces an activity set has to occur in.
        materialize: Run a reasoner after writing.
        reasoner_backend: Reasoner for `materialize`.
        save_backmap: Write the maps of cleaned names to source names, one per
            kind of name.
        streaming: Write functional syntax to disk as the build runs. Off
            builds in memory and writes RDF/XML.
        output_formats: Further formats to write each ontology in.
        size_limit_mb: Skip inputs larger than this. Zero for no limit.
        log_dir: Directory for per-input log files.
        warnings: Warnings to log before the run starts.
    """

    inputs: tuple[Path, ...]
    output_dir: Path
    name: str | None
    settings: OntoBuildSettings
    case_concepts: tuple[str, ...] | None
    iri: str | None
    auto_case_concepts: int
    case_col: str | None
    act_col: str | None
    time_col: str | None
    object_cols: list[str]
    object_attribute_cols: list[str]
    annotation_document: str | None
    declare_file: str | None
    constraints: list[str] | None
    constraint_support: float
    itemset_support: float
    materialize: bool
    reasoner_backend: ReasonerBackend
    save_backmap: bool
    streaming: bool
    output_formats: list[OutputFormat]
    size_limit_mb: int
    log_dir: Path | None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def output_stem(self, input_path: Path) -> str:
        """Return the shared name of the artifacts built from one input."""
        return self.name or input_path.stem

    def log_iri(self, input_path: Path) -> str:
        """Return the base IRI of the names from one input."""
        return self.iri or default_iri(self.output_stem(input_path))

    def ontology_path(self, input_path: Path) -> Path:
        """Return where the ontology for one input goes.

        `.ofn` for a streamed build, `.owl` for an in-memory one.
        """
        return (
            self.output_dir
            / f"{self.output_stem(input_path)}.{self.build_format.value}"
        )

    @property
    def build_format(self) -> OutputFormat:
        """The format the build itself writes."""
        return OutputFormat.OFN if self.streaming else OutputFormat.OWL

    def converted_paths(self, input_path: Path) -> dict[OutputFormat, Path]:
        """Return where each further format of one input goes.

        The format the build writes itself is left out, and a repeated format
        appears once.
        """
        stem = self.output_stem(input_path)
        return {
            f: self.output_dir / f"{stem}.{f.value}"
            for f in self.output_formats
            if f is not self.build_format
        }

    def backmap_path(self, input_path: Path) -> Path:
        """Return where the name backmap for one input goes."""
        return self.output_dir / f"{self.output_stem(input_path)}.backmap.json"

    def declare_result(self) -> DeclareResult | None:
        """Read the mined constraints, or None when the run names no file.

        Raises:
            ConfigError: If the file is not there.
        """
        if self.declare_file is None:
            return None
        if not Path(self.declare_file).is_file():
            raise ConfigError(
                f"declare_file not found: {self.declare_file}. Mine the "
                "constraints first with 'semlog mine-declare'."
            )
        return DeclareResult.read(self.declare_file)

    def _annotations(self) -> dict[str, dict[str, str]]:
        """Read the annotation document, or return {} when none is named.

        Raises:
            ConfigError: If the file is not there, is not JSON, or is not an
                object of entity name to {label, comment}.
        """
        if self.annotation_document is None:
            return {}
        path = Path(self.annotation_document)
        if not path.is_file():
            raise ConfigError(
                f"annotation_document not found: {path}. Point it at an existing "
                "JSON file."
            )
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}: invalid JSON: {exc}.") from exc
        problem = _annotation_document_problem(document)
        if problem is not None:
            raise ConfigError(f"{path}: {problem}")
        return document

    def build_inputs(
        self, case_concepts: Sequence[str], declare: DeclareResult | None
    ) -> BuildInputs:
        """Return the `BuildInputs` for one log.

        Reads the annotation document when the run names one.

        Args:
            case_concepts: Object types this input resolved to.
            declare: The mined constraints, as `declare_result` read them.

        Raises:
            ConfigError: If the annotation document is missing or malformed.
        """
        return BuildInputs(
            case_concepts=tuple(case_concepts),
            declare=declare,
            annotations=self._annotations(),
        )

    def loader_columns(self) -> dict[str, Any]:
        """Return the column names a flat log is read with."""
        return _loader_columns({key: getattr(self, key) for key in COLUMN_KEYS})

    def value_of(self, declared: Setting) -> Any:
        """Return what one setting resolved to on this run."""
        holder = self.settings if declared.target == "settings" else self
        return getattr(holder, declared.field or declared.key)

    def render(self) -> str:
        """Return the run as TOML, as printed by --dry-run.

        Every set setting is written, with a directory input expanded to its
        files. An unset setting is omitted. The output paths follow as
        comments.
        """
        lines = ["# The settings this run resolved to.", ""]
        for declared in BUILD_SETTINGS:
            value = self.value_of(declared)
            if value is not None:
                lines.append(f"{declared.key} = {toml_value(value)}")

        lines += ["", "# would write:"]
        for path in self.inputs:
            lines.append(f"#   {self.ontology_path(path)}")
            lines.extend(f"#   {p}" for p in self.converted_paths(path).values())
            if self.save_backmap:
                lines.append(f"#   {self.backmap_path(path)}")
        return "\n".join(lines)


def _annotation_document_problem(document: Any) -> str | None:
    """Return what is wrong with an annotation document, or None if nothing."""
    if not isinstance(document, dict):
        return "expected a JSON object of entity name to {label, comment}."
    for name, entry in document.items():
        if not isinstance(entry, dict):
            return f"entry {name!r}: expected an object with label and comment."
        unknown = sorted(set(entry) - _ANNOTATION_KEYS)
        if unknown:
            return f"entry {name!r}: unknown key(s) {unknown}. Use label and comment."
        for key, value in entry.items():
            if not isinstance(value, str):
                return f"entry {name!r}: {key} must be a string, got {value!r}."
    return None


def _settings_from(
    merged: dict[str, Any], flagged: Collection[str]
) -> OntoBuildSettings:
    """Build the ontology settings from the merged configuration."""
    kwargs: dict[str, Any] = {
        declared.key: read_setting(merged, declared.key, flagged)
        for declared in settings_targeting("settings")
    }
    return OntoBuildSettings(**kwargs)


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
        "attribute_representations (currently "
        f"{[r.value for r in settings.attribute_representations]})."
    ]


def _column_warnings(merged: dict[str, Any], inputs: Sequence[Path]) -> list[str]:
    """Warn about column settings on a run with no flat log to read them."""
    named = [key for key in COLUMN_KEYS if merged[key] != BUILD_DEFAULTS[key]]
    if not named:
        return []
    if any(path.suffix.lower() in CLASSICAL_SUFFIXES for path in inputs):
        return []
    return [
        f"{', '.join(named)} name columns of a flat .csv or .xes log and have "
        "no effect on the OCEL inputs of this run."
    ]


def resolve_run_config(
    overrides: dict[str, Any],
    *,
    config_file: str | Path | None = None,
) -> RunConfig:
    """Resolve defaults, config file and flags into one checked run.

    Settings are validated without the per-log inputs, such as case concepts
    chosen from the data. The builder validates again with them.

    Args:
        overrides: Settings given on the command line.
        config_file: TOML config to read first.

    Raises:
        ConfigError: If a setting is missing, unknown, of the wrong sort, or
            contradicts another one.
    """
    merged = _merged_settings(overrides, config_file)

    settings = _settings_from(merged, overrides)
    run: dict[str, Any] = {
        s.field or s.key: read_setting(merged, s.key, overrides)
        for s in settings_targeting("run")
    }
    inputs: tuple[Path, ...] = run["inputs"]

    settings.validate()

    if settings.emits_declare and run["declare_file"] is None:
        raise ConfigError(
            "order_representations includes 'declare' but no declare_file is "
            "set. Mine the constraints first with 'semlog mine-declare', then "
            "point declare_file at the file it writes."
        )

    if run["name"] is not None:
        if len(inputs) > 1:
            raise ConfigError(
                f"name={run['name']!r} would make {len(inputs)} inputs share one "
                "output file. Drop it, or build one input at a time."
            )
    elif config_file is not None and len(inputs) == 1:
        run["name"] = Path(config_file).stem

    if run["case_concepts"] == () and settings.emits_traces:
        raise ConfigError(
            "case_concepts is empty but order_representations needs traces "
            f"({', '.join(r.value for r in settings.order_representations)}). "
            "Name the object types to use as cases, or drop them from "
            "order_representations."
        )

    return RunConfig(
        settings=settings,
        warnings=tuple(
            _binning_warnings(merged, settings) + _column_warnings(merged, inputs)
        ),
        **run,
    )
