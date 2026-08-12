"""Turning defaults, a TOML file and flags into one checked configuration.

Precedence runs defaults, then the config file, then the flags. Everything is
resolved and validated here, so a run either has a `RunConfig` that describes
a build that can work, or it has failed with a message about the setting that
does not.

Configs can extend one another, which is how several runs over one dataset
share their common settings.
"""

from __future__ import annotations

import difflib
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from semanticlog.data.loading.pm4py_loader import OCEL_SUFFIXES
from semanticlog.onto.settings import (
    DEFAULT_BINS,
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OCELExtension,
    OntoBuildSettings,
    OWLProfile,
)


class ConfigError(ValueError):
    """A setting is missing, unknown, or of the wrong sort."""


def parse_enum[E: Enum](enum_cls: type[E], token: str, *, key: str) -> E:
    """Read an enum member from a written setting.

    Both spellings are accepted, case-insensitively, so `--order-representations DF` and
    `--order-representations df` mean the same thing.

    Args:
        enum_cls: Enum to read from.
        token: Text to read.
        key: Setting name, quoted in the error.

    Returns:
        The member the text names.

    Raises:
        ConfigError: If nothing matches, listing what would.
    """
    folded = token.strip().casefold()
    for member in enum_cls:
        value = member.value
        if folded in (str(value).casefold(), member.name.casefold()):
            return member
    allowed = ", ".join(str(m.value) for m in enum_cls)
    raise ConfigError(f"{key}: unknown value {token!r}. Choose from: {allowed}.")


def enum_choices(enum_cls: type[Enum]) -> list[str]:
    """List the spellings of an enum, for help text."""
    values = [str(m.value) for m in enum_cls]
    names = [m.name for m in enum_cls if m.name not in values]
    return values + names


BUILD_DEFAULTS: dict[str, Any] = {
    "input": None,
    "output": "datasets/onto",
    "name": None,
    "profile": OWLProfile.OWL2DL.value,
    "order_representations": [EventOrderRepresentation.DF.value],
    "case_concepts": None,
    "auto_case_concepts": 3,
    "inverse_representations": [],
    "ocel_extensions": [],
    "data_representations": [DataRepresentation.VALUES.value],
    "bins": DEFAULT_BINS,
    "streaming": False,
    "bin_labels": None,
    "annotation_document": None,
    "derive_annotations": False,
    "declare_file": None,
    "materialize": False,
    "reasoner_backend": "HermiT",
    "save_backmap": False,
    "save_store": False,
    "size_limit_mb": 0,
    "log_dir": None,
}


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
        save_store: Also write the loaded store.
        streaming: Write the ontology straight to disk.
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
    save_store: bool = False
    streaming: bool = False
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

    def store_path(self, input_path: Path) -> Path:
        """Return where the loaded store for one input goes."""
        return self.output_dir / f"{self.output_stem(input_path)}.oced"

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
        if DataRepresentation.EQUAL_BINS in settings.data_representations or (
            DataRepresentation.DIST_BINS in settings.data_representations
        ):
            lines.append(
                f"bins                 {settings.bins} ({', '.join(settings.bin_labels)})"
            )
        lines.append("")
        lines.append("would write:")
        for path in self.inputs:
            lines.append(f"  {self.ontology_path(path)}")
            if self.save_backmap:
                lines.append(f"  {self.backmap_path(path)}")
            if self.save_store:
                lines.append(f"  {self.store_path(path)}")
        return "\n".join(lines)


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


def _expand_inputs(tokens: Sequence[str]) -> tuple[Path, ...]:
    """Turn the input setting into the actual list of files to build.

    A directory contributes every OCEL file in it. A path that does not exist
    raises rather than yielding an empty result.
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
    """Build the ontology settings from the merged configuration."""
    bins = _as_int(merged["bins"], "bins")
    bin_labels = merged["bin_labels"]
    kwargs: dict[str, Any] = {
        "profile": parse_enum(
            OWLProfile, _as_str(merged["profile"], "profile"), key="profile"
        ),
        "order_representations": [
            parse_enum(
                EventOrderRepresentation,
                _as_str(v, "order_representations"),
                key="order_representations",
            )
            for v in _as_list(merged["order_representations"])
        ],
        "ocel_extensions": [
            parse_enum(
                OCELExtension, _as_str(v, "ocel_extensions"), key="ocel_extensions"
            )
            for v in _as_list(merged["ocel_extensions"])
        ],
        "inverse_representations": [
            parse_enum(
                InverseRepresentation,
                _as_str(v, "inverse_representations"),
                key="inverse_representations",
            )
            for v in _as_list(merged["inverse_representations"])
        ],
        "data_representations": [
            parse_enum(
                DataRepresentation,
                _as_str(v, "data_representations"),
                key="data_representations",
            )
            for v in _as_list(merged["data_representations"])
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
        kwargs["bin_labels"] = [_as_str(v, "bin_labels") for v in _as_list(bin_labels)]

    try:
        return OntoBuildSettings(**kwargs)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _binning_warnings(merged: dict[str, Any], settings: OntoBuildSettings) -> list[str]:
    """Warn about bin settings that were given but will not be used."""
    binning = {DataRepresentation.EQUAL_BINS, DataRepresentation.DIST_BINS}
    if binning & set(settings.data_representations):
        return []
    if merged["bins"] == BUILD_DEFAULTS["bins"] and merged["bin_labels"] is None:
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

    inputs = _expand_inputs(_as_list(merged["input"]))
    settings = _settings_from(merged)

    try:
        settings.validate(require_inputs=False)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    if _as_bool(merged["streaming"], "streaming") and _as_bool(
        merged["materialize"], "materialize"
    ):
        raise ConfigError(
            "streaming and materialize cannot both be set: a streamed build "
            "keeps no ontology in memory for the reasoner to read. Run "
            "'semlog reason' on the written file instead."
        )

    if settings.emits_declare:
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
            _as_str(v, "case_concepts") for v in _as_list(case_concepts)
        )
        if not case_concepts and settings.emits_traces:
            raise ConfigError(
                "case_concepts is empty but order needs traces "
                f"({', '.join(r.value for r in settings.order_representations)}). "
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
        streaming=_as_bool(merged["streaming"], "streaming"),
        size_limit_mb=_as_int(merged["size_limit_mb"], "size_limit_mb"),
        log_dir=None if merged["log_dir"] is None else Path(merged["log_dir"]),
        config_file=None if config_file is None else Path(config_file),
        command=command,
        warnings=tuple(_binning_warnings(merged, settings)),
    )
