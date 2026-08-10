"""The resolved configuration of a single CLI run.

A :class:`RunConfig` is the only value the pipeline in :mod:`semanticlog.cli.run`
consumes.  TOML files and command line flags are both funnelled into one here,
so no pipeline code ever sees an :class:`argparse.Namespace`.

Precedence is flag > config file > built-in default.  Flags default to ``None``
in the parser so that "not given" stays distinguishable from "given the value
that happens to be the default".

Relative paths inside a config file are resolved against the current working
directory, not against the config file, so that a config reads the same way as
the equivalent command line typed from the project root.  ``extends`` is the one
exception: it names another config and resolves relative to the file it appears
in.
"""

from __future__ import annotations

import copy
import difflib
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from semanticlog.data.loading.pm4pyloader import OCEL_SUFFIXES
from semanticlog.onto.settings import (
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OcelExtension,
    OntoBuildSettings,
    OwlProfile,
)


class ConfigError(ValueError):
    """A run configuration that cannot be resolved into a :class:`RunConfig`."""


# ---------------------------------------------------------------------------
# Enum tokens
# ---------------------------------------------------------------------------


def parse_enum[E: Enum](enum_cls: type[E], token: str, *, key: str) -> E:
    """Resolve ``token`` to a member of ``enum_cls`` by value or by name.

    Both spellings are accepted case-insensitively, so ``DF``, ``df``,
    ``EF_explicit`` and ``ef_explicit`` all work.  The enums mix the two
    conventions (``OcelExtension.ATTRIBUTE_HISTORY`` has the value
    ``"attribute_history"``), and a user should not have to know which is which.

    Args:
        enum_cls: The enum to resolve against.
        token: The spelling given on the command line or in a config file.
        key: Setting name, used only to make the error message actionable.

    Raises:
        ConfigError: If ``token`` matches neither a value nor a name.
    """
    folded = token.strip().casefold()
    for member in enum_cls:
        value = member.value
        if folded in (str(value).casefold(), member.name.casefold()):
            return member
    allowed = ", ".join(str(m.value) for m in enum_cls)
    raise ConfigError(f"{key}: unknown value {token!r}. Choose from: {allowed}.")


def enum_choices(enum_cls: type[Enum]) -> list[str]:
    """Every accepted spelling for ``enum_cls``, values first then distinct names."""
    values = [str(m.value) for m in enum_cls]
    names = [m.name for m in enum_cls if m.name not in values]
    return values + names


# ---------------------------------------------------------------------------
# Config keys
# ---------------------------------------------------------------------------

#: Every key a build config file may contain, with the value used when neither
#: the file nor a flag supplies one.
BUILD_DEFAULTS: dict[str, Any] = {
    "input": None,
    "output": "data/onto",
    "name": None,
    "profile": OwlProfile.OWL2DL.value,
    "order": [EventOrderRepresentation.DF.value],
    "case_concepts": None,
    "auto_case_concepts": 3,
    "inverse_representation": [],
    "ocel_extensions": [],
    "data_representation": [DataRepresentation.VALUES.value],
    "bins": 5,
    "bin_labels": None,
    "annotation_document": None,
    "derive_annotations": False,
    "declare_file": None,
    "materialize": False,
    "reasoner_backend": "HermiT",
    "save_backmap": False,
    "save_store": False,
    "size_limit": 0,
    "log_dir": None,
}


def _check_keys(raw: dict[str, Any], source: Path) -> None:
    unknown = sorted(set(raw) - set(BUILD_DEFAULTS))
    if not unknown:
        return
    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key, BUILD_DEFAULTS, n=1)
        hints.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    raise ConfigError(
        f"{source}: unknown setting(s): {', '.join(hints)}.\n"
        f"Valid settings: {', '.join(sorted(BUILD_DEFAULTS))}."
    )


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Read a build config, following ``extends`` chains.

    Args:
        path: Path to a TOML config file.

    Returns:
        A flat dict of settings, with each file overriding the one it extends.

    Raises:
        ConfigError: If the file is missing, malformed, contains an unknown
            setting, or its ``extends`` chain forms a cycle.
    """
    return _load_config_file(Path(path), [])


def _load_config_file(path: Path, chain: list[Path]) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# The resolved config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """Everything a build needs, with every value already resolved and checked.

    Attributes:
        inputs: Existing input files, in the order they will be processed.
        output_dir: Directory the ontology and manifest are written to.
        name: Output stem shared by all artifacts, or ``None`` to derive it from
            each input file name.
        settings: Build settings, minus ``case_concepts``, which is decided per
            input file because it may be auto-selected.
        case_concepts: Explicit case concepts, or ``None`` to auto-select.
        auto_case_concepts: How many object types to auto-select when needed.
        materialize: Run the reasoner over each ontology after writing it.
        reasoner_backend: Reasoner used when ``materialize`` is set.
        save_backmap: Also write the cleaned-name to original-name map.
        save_store: Also write the loaded OCEDStore as a pickle.
        size_limit_mb: Skip inputs larger than this, or 0 for no limit.
        log_dir: Directory for per-input log files, or ``None``.
        config_file: The config this run was resolved from, if any.
        command: The command line that produced this config, for the manifest.
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
    save_store: bool = False
    size_limit_mb: int = 0
    log_dir: Path | None = None
    config_file: Path | None = None
    command: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def output_stem(self, input_path: Path) -> str:
        """The shared stem for every artifact produced from ``input_path``."""
        return self.name or input_path.stem

    def ontology_path(self, input_path: Path) -> Path:
        return self.output_dir / f"{self.output_stem(input_path)}.owl"

    def manifest_path(self, input_path: Path) -> Path:
        return self.output_dir / f"{self.output_stem(input_path)}.manifest.json"

    def backmap_path(self, input_path: Path) -> Path:
        return self.output_dir / f"{self.output_stem(input_path)}.backmap.json"

    def store_path(self, input_path: Path) -> Path:
        # A directory, not a file: the store is Parquet tables plus a manifest.
        return self.output_dir / f"{self.output_stem(input_path)}.oced"

    def settings_for(self, case_concepts: Sequence[str]) -> OntoBuildSettings:
        """A copy of ``settings`` carrying the case concepts chosen for one input.

        The copy is shallow on purpose: the lazily loaded DECLARE and annotation
        data are read-only and worth sharing across the files of a batch.
        """
        settings = copy.copy(self.settings)
        settings.case_concepts = list(case_concepts)
        return settings


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _as_list(value: Any, key: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Iterable):
        return [value]
    return list(value)


def _as_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{key}: expected true or false, got {value!r}.")


def _as_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key}: expected a whole number, got {value!r}.")
    return value


def _as_str(value: Any, key: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{key}: expected a string, got {value!r}.")
    return value


def _expand_inputs(tokens: Sequence[str]) -> tuple[Path, ...]:
    """Turn the given file and directory paths into a concrete list of inputs.

    A directory contributes every file in it whose suffix the OCEL loader
    understands.  A file named explicitly is always taken, so an unusual
    extension can still be forced.  A path that does not exist is an error
    rather than a silent omission.
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
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return tuple(unique)


def _settings_from(merged: dict[str, Any]) -> OntoBuildSettings:
    bins = _as_int(merged["bins"], "bins")
    bin_labels = merged["bin_labels"]
    kwargs: dict[str, Any] = {
        "profile": parse_enum(
            OwlProfile, _as_str(merged["profile"], "profile"), key="profile"
        ),
        "order_representation": [
            parse_enum(EventOrderRepresentation, _as_str(v, "order"), key="order")
            for v in _as_list(merged["order"], "order")
        ],
        "ocel_extensions": [
            parse_enum(
                OcelExtension, _as_str(v, "ocel_extensions"), key="ocel_extensions"
            )
            for v in _as_list(merged["ocel_extensions"], "ocel_extensions")
        ],
        "inverse_representation": [
            parse_enum(
                InverseRepresentation,
                _as_str(v, "inverse_representation"),
                key="inverse_representation",
            )
            for v in _as_list(
                merged["inverse_representation"], "inverse_representation"
            )
        ],
        "data_representation": [
            parse_enum(
                DataRepresentation,
                _as_str(v, "data_representation"),
                key="data_representation",
            )
            for v in _as_list(merged["data_representation"], "data_representation")
        ],
        "bins": bins,
        "annotation_document": (
            None
            if merged["annotation_document"] is None
            else _as_str(merged["annotation_document"], "annotation_document")
        ),
        "declare_file": (
            None
            if merged["declare_file"] is None
            else _as_str(merged["declare_file"], "declare_file")
        ),
        "derive_annotations": _as_bool(
            merged["derive_annotations"], "derive_annotations"
        ),
    }
    if bin_labels is not None:
        kwargs["bin_labels"] = [
            _as_str(v, "bin_labels") for v in _as_list(bin_labels, "bin_labels")
        ]

    try:
        return OntoBuildSettings(**kwargs)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _binning_warnings(merged: dict[str, Any], settings: OntoBuildSettings) -> list[str]:
    binning = {DataRepresentation.EQUAL_BINS, DataRepresentation.DIST_BINS}
    if binning & set(settings.data_representation):
        return []
    if merged["bins"] == BUILD_DEFAULTS["bins"] and merged["bin_labels"] is None:
        return []
    return [
        "bins/bin_labels have no effect without equal_bins or dist_bins in "
        f"data_representation (currently {[r.value for r in settings.data_representation]})."
    ]


def resolve_run_config(
    overrides: dict[str, Any],
    *,
    config_file: str | Path | None = None,
    command: str = "",
) -> RunConfig:
    """Merge a config file with command line overrides into a :class:`RunConfig`.

    Args:
        overrides: Settings given explicitly on the command line. Keys absent
            here fall back to the config file and then to
            :data:`BUILD_DEFAULTS`.
        config_file: Path to a TOML config, or ``None`` for flags only.
        command: The command line as typed, recorded in the run manifest.

    Raises:
        ConfigError: If the merged configuration is unusable.
    """
    from_file = load_config_file(config_file) if config_file is not None else {}
    merged: dict[str, Any] = {**BUILD_DEFAULTS, **from_file, **overrides}

    inputs = _expand_inputs(_as_list(merged["input"], "input"))
    settings = _settings_from(merged)

    try:
        settings.validate(require_inputs=False)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    if settings.instantiate_declare:
        if settings.declare_file is None:
            raise ConfigError(
                "order includes 'declare' but no declare_file is set. Mine the "
                "constraints first with 'semlog mine-declare', then point "
                "declare_file at the CSV it writes."
            )
        if not Path(settings.declare_file).is_file():
            raise ConfigError(
                f"declare_file not found: {settings.declare_file}. Mine the "
                "constraints first with 'semlog mine-declare'."
            )

    if (
        settings.annotation_document is not None
        and not Path(settings.annotation_document).is_file()
    ):
        raise ConfigError(
            f"annotation_document not found: {settings.annotation_document}"
        )

    name = merged["name"]
    if name is not None:
        name = _as_str(name, "name")
        if len(inputs) > 1:
            raise ConfigError(
                f"name={name!r} would make {len(inputs)} inputs share one output "
                "file. Drop it, or build one input at a time."
            )
    elif config_file is not None and len(inputs) == 1:
        name = Path(config_file).stem

    case_concepts = merged["case_concepts"]
    if case_concepts is not None:
        case_concepts = tuple(
            _as_str(v, "case_concepts")
            for v in _as_list(case_concepts, "case_concepts")
        )
        if not case_concepts and settings.instantiate_traces:
            raise ConfigError(
                "case_concepts is empty but order needs traces "
                f"({', '.join(r.value for r in settings.order_representation)}). "
                "Name the object types to use as cases, or drop them from order."
            )

    return RunConfig(
        inputs=inputs,
        output_dir=Path(_as_str(merged["output"], "output")),
        name=name,
        settings=settings,
        case_concepts=case_concepts,
        auto_case_concepts=_as_int(merged["auto_case_concepts"], "auto_case_concepts"),
        materialize=_as_bool(merged["materialize"], "materialize"),
        reasoner_backend=_as_str(merged["reasoner_backend"], "reasoner_backend"),
        save_backmap=_as_bool(merged["save_backmap"], "save_backmap"),
        save_store=_as_bool(merged["save_store"], "save_store"),
        size_limit_mb=_as_int(merged["size_limit"], "size_limit"),
        log_dir=None if merged["log_dir"] is None else Path(merged["log_dir"]),
        config_file=None if config_file is None else Path(config_file),
        command=command,
        warnings=tuple(_binning_warnings(merged, settings)),
    )


def describe(cfg: RunConfig) -> str:
    """Render a resolved config as the text ``--dry-run`` prints."""
    settings = cfg.settings
    case_concepts = (
        f"auto (the {cfg.auto_case_concepts} most common object types)"
        if cfg.case_concepts is None
        else (", ".join(cfg.case_concepts) or "none")
    )
    lines = [
        f"config file          {cfg.config_file or '(none, flags only)'}",
        f"inputs               {len(cfg.inputs)}",
        *(f"  - {path}" for path in cfg.inputs),
        f"output directory     {cfg.output_dir}",
        f"profile              {settings.profile.value}",
        f"order                {', '.join(r.value for r in settings.order_representation) or 'none'}",
        f"case concepts        {case_concepts}",
        f"data representation  {', '.join(r.value for r in settings.data_representation) or 'none'}",
        f"inverses             {', '.join(r.value for r in settings.inverse_representation) or 'none'}",
        f"ocel extensions      {', '.join(e.value for e in settings.ocel_extensions) or 'none'}",
        f"annotations          document={settings.annotation_document or 'none'} derived={settings.derive_annotations}",
        f"declare constraints  {settings.declare_file or 'none'}",
        f"materialize          {cfg.materialize}"
        + (f" ({cfg.reasoner_backend})" if cfg.materialize else ""),
    ]
    if DataRepresentation.EQUAL_BINS in settings.data_representation or (
        DataRepresentation.DIST_BINS in settings.data_representation
    ):
        lines.append(
            f"bins                 {settings.bins} ({', '.join(settings.bin_labels)})"
        )
    lines.append("")
    lines.append("would write:")
    for path in cfg.inputs:
        lines.append(f"  {cfg.ontology_path(path)}")
        lines.append(f"  {cfg.manifest_path(path)}")
        if cfg.save_backmap:
            lines.append(f"  {cfg.backmap_path(path)}")
        if cfg.save_store:
            lines.append(f"  {cfg.store_path(path)}")
    return "\n".join(lines)
