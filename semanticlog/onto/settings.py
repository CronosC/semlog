"""What a build emits, as one checked value.

`OntoBuildSettings` is the single configuration object of a build. It is
frozen, it validates itself, and it resolves the files it points at when it is
constructed, so a builder never has to check whether an input is loaded yet.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from semanticlog.core.enums import (
    BuildEnum,
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OCELExtension,
    OWLProfile,
)
from semanticlog.data.binning import DEFAULT_BIN_LABELS, DEFAULT_BINS
from semanticlog.enrich.declare import DeclareResult

DEFAULT_MAX_EF_PAIRS = 50_000_000


def _read_annotations(path: str) -> dict[str, dict[str, str]]:
    """Read an annotation document, which maps a name to its label and comment."""
    with open(path, encoding="utf-8") as f:
        document: dict[str, dict[str, str]] = json.load(f)
    return document


@dataclass(frozen=True)
class FileBackedSetting:
    """A setting that names a file, paired with the field holding its content.

    Everything that knows about one of these reads it from `FILE_BACKED`: the
    settings resolve it on construction, `replace` drops stale content when the
    path changes, and the CLI checks the file exists before either happens.

    Attributes:
        path_field: Field naming the file.
        content_field: Field the loaded content lands on.
        read: Turns a path into that content.
        missing: What to say when the file is not there, with `{path}` in it.
        copy_content: Deep-copy content passed directly, so a caller cannot
            mutate it afterwards. Off for content that is already immutable or
            too large to copy for nothing.
    """

    path_field: str
    content_field: str
    read: Callable[[str], Any]
    missing: str
    copy_content: bool = False


FILE_BACKED: tuple[FileBackedSetting, ...] = (
    FileBackedSetting(
        "declare_file",
        "declare_result",
        lambda path: DeclareResult.read(path),
        "declare_file not found: {path}. Mine the constraints first with "
        "'semlog mine-declare'.",
    ),
    FileBackedSetting(
        "annotation_document",
        "annotation_data",
        _read_annotations,
        "annotation_document not found: {path}",
        copy_content=True,
    ),
)


@dataclass(frozen=True)
class OntoBuildSettings:
    """Everything a build needs to know beyond the data itself.

    Sequence fields accept any iterable and are stored as tuples. The two
    file-backed fields are read once, when the settings are constructed, and
    the loaded content is then available as `annotation_data` and
    `declare_result`.

    Attributes:
        profile: OWL 2 profile the output stays inside.
        order_representations: How event order is expressed. Empty for none.
        ocel_extensions: Optional OCEL 2.0 features to emit.
        case_concepts: Object types whose instances act as cases. Required by
            every order representation.
        declare_file: Path to a mined DECLARE constraints CSV.
        inverse_representations: How inverse properties are expressed. Empty
            for no inverses.
        data_representations: How numeric attribute values are expressed.
        bins: Number of bins for the binning modes.
        bin_labels: Label per bin. Must have exactly `bins` entries.
        annotation_document: Path to a JSON document of entity name to label
            and comment.
        derive_annotations: Emit labels and comments derived from the original
            names in the log.
        max_ef_pairs: Ceiling on materialized eventually-follows pairs, which
            grow quadratically with trace length. Zero for no limit.
        annotation_data: Loaded content of `annotation_document`, or annotations
            passed directly.
        declare_result: Loaded content of `declare_file`, or constraints passed
            directly.

    Raises:
        ValueError: If `bin_labels` does not have exactly `bins` entries.
    """

    profile: OWLProfile = OWLProfile.OWL2DL
    order_representations: tuple[EventOrderRepresentation, ...] = ()
    ocel_extensions: tuple[OCELExtension, ...] = ()
    case_concepts: tuple[str, ...] = ()
    declare_file: str | None = None
    inverse_representations: tuple[InverseRepresentation, ...] = ()
    data_representations: tuple[DataRepresentation, ...] = (DataRepresentation.VALUES,)
    bins: int = DEFAULT_BINS
    bin_labels: tuple[str, ...] = tuple(DEFAULT_BIN_LABELS)
    annotation_document: str | None = None
    derive_annotations: bool = False
    max_ef_pairs: int = DEFAULT_MAX_EF_PAIRS
    annotation_data: dict[str, dict[str, str]] | None = None
    declare_result: DeclareResult | None = None

    _SEQUENCES: ClassVar[tuple[str, ...]] = (
        "order_representations",
        "ocel_extensions",
        "case_concepts",
        "inverse_representations",
        "data_representations",
        "bin_labels",
    )

    _ENUM_FIELDS: ClassVar[dict[str, type[BuildEnum]]] = {
        "profile": OWLProfile,
        "order_representations": EventOrderRepresentation,
        "ocel_extensions": OCELExtension,
        "inverse_representations": InverseRepresentation,
        "data_representations": DataRepresentation,
    }

    def __post_init__(self) -> None:
        for name in self._SEQUENCES:
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))
        self._resolve_enums()
        self._resolve_inputs()
        problem = self._bin_label_problem()
        if problem is not None:
            raise ValueError(problem)

    def _resolve_enums(self) -> None:
        """Turn whatever was passed for an enum field into the member itself.

        These enums subclass `str`, so a bare string compares equal to a member
        and behaves correctly everywhere except where the member itself is
        wanted. Coercing here means a misspelling is a message about the
        setting rather than an AttributeError from somewhere later.
        """
        for name, enum_cls in self._ENUM_FIELDS.items():
            value = getattr(self, name)
            if isinstance(value, tuple):
                resolved: Any = tuple(enum_cls.parse(v, key=name) for v in value)
            else:
                resolved = enum_cls.parse(value, key=name)
            object.__setattr__(self, name, resolved)

    def _resolve_inputs(self) -> None:
        """Read the file-backed fields, and copy content passed directly."""
        for setting in FILE_BACKED:
            content = getattr(self, setting.content_field)
            path = getattr(self, setting.path_field)
            if content is None and path is not None:
                object.__setattr__(self, setting.content_field, setting.read(path))
            elif content is not None and setting.copy_content:
                object.__setattr__(self, setting.content_field, copy.deepcopy(content))

    def _bin_label_problem(self) -> str | None:
        if len(self.bin_labels) != self.bins:
            return (
                f"bin_labels must have exactly bins={self.bins} "
                f"elements, got {len(self.bin_labels)}."
            )
        return None

    def replace(self, **changes: Any) -> OntoBuildSettings:
        """Return a copy with some fields changed.

        Changing a path field drops the content loaded from the old path, so
        the new one is read instead of the stale content being kept.

        Args:
            **changes: Field values to change.

        Returns:
            A new settings object.
        """
        for setting in FILE_BACKED:
            if setting.path_field in changes and setting.content_field not in changes:
                changes[setting.content_field] = None
        return dataclasses.replace(self, **changes)

    @property
    def has_inverses(self) -> bool:
        """Whether any inverse representation was selected."""
        return bool(self.inverse_representations)

    @property
    def axiomatic_inverses(self) -> bool:
        """Whether inverses are related by an axiom."""
        return InverseRepresentation.AXIOMATIC in self.inverse_representations

    @property
    def implicit_inverses(self) -> bool:
        """Whether inverse pairs are asserted directly."""
        return InverseRepresentation.IMPLICIT in self.inverse_representations

    @property
    def emits_ef(self) -> bool:
        """Whether eventually-follows is emitted in either form."""
        return self.emits_ef_axiomatic or self.emits_ef_explicit

    @property
    def emits_ef_axiomatic(self) -> bool:
        """Whether eventually-follows is left to a reasoner."""
        return EventOrderRepresentation.EF in self.order_representations

    @property
    def emits_ef_explicit(self) -> bool:
        """Whether every eventually-follows pair is asserted."""
        return EventOrderRepresentation.EF_EXPLICIT in self.order_representations

    @property
    def emits_df(self) -> bool:
        """Whether directly-follows properties and their pairs are emitted.

        EF derives its pairs by transitivity over directly-follows, so a build
        that leaves them to a reasoner emits directly-follows as well.
        """
        return (
            EventOrderRepresentation.DF in self.order_representations
            or self.emits_ef_axiomatic
        )

    @property
    def emits_declare(self) -> bool:
        """Whether traces are typed by the DECLARE constraints they satisfy."""
        return EventOrderRepresentation.DECLARE in self.order_representations

    @property
    def emits_traces(self) -> bool:
        """Whether the log has to be projected onto its case concepts."""
        return self.emits_df or self.emits_ef

    @property
    def emits_attribute_history(self) -> bool:
        """Whether object attribute changes are emitted."""
        return OCELExtension.ATTRIBUTE_HISTORY in self.ocel_extensions

    @property
    def binning_mode(self) -> DataRepresentation | None:
        """The selected binning mode, or None if numeric values are not binned."""
        for mode in (DataRepresentation.EQUAL_BINS, DataRepresentation.DIST_BINS):
            if mode in self.data_representations:
                return mode
        return None

    def validate(self, require_inputs: bool = True) -> None:
        """Check that the settings describe a build that can run.

        Every problem is collected before raising, so one call reports all of
        them.

        Args:
            require_inputs: Also check the settings that depend on data the
                build has not seen yet, such as case concepts and DECLARE
                constraints. The CLI resolves its configuration before those
                are known, so it passes False there and the builder checks
                again with True.

        Raises:
            ValueError: If any setting is unusable, listing every problem.
        """
        errors: list[str] = []
        is_el = self.profile == OWLProfile.OWL2EL

        if EventOrderRepresentation.EF in self.order_representations and is_el:
            errors.append(
                "EventOrderRepresentation.EF leaves the pairs to a reasoner, and ELK, "
                "which is the reasoner for OWL2EL, does not answer for property "
                "assertions, so no pair would ever be derived. Use "
                "EventOrderRepresentation.EF_EXPLICIT instead, or switch to "
                "OWLProfile.OWL2DL and reason with HermiT or Pellet."
            )

        if (
            self.emits_ef_axiomatic
            and EventOrderRepresentation.DF not in self.order_representations
        ):
            errors.append(
                "EventOrderRepresentation.EF derives its pairs by transitivity over "
                "directly-follows, so with no EventOrderRepresentation.DF in "
                "order_representations there is nothing for a reasoner to derive them "
                "from. Add DF, or use EventOrderRepresentation.EF_EXPLICIT to assert "
                "every pair without a reasoner."
            )

        if self.axiomatic_inverses and is_el:
            errors.append(
                "InverseRepresentation.AXIOMATIC (InverseObjectProperty TBox axioms) is not "
                "supported in OWL2EL. Remove AXIOMATIC from inverse_representations or switch "
                "to OWLProfile.OWL2DL."
            )

        if (
            DataRepresentation.EQUAL_BINS in self.data_representations
            and DataRepresentation.DIST_BINS in self.data_representations
        ):
            errors.append(
                "DataRepresentation.EQUAL_BINS and DIST_BINS are mutually exclusive; "
                "include at most one in data_representations."
            )

        if require_inputs:
            if self.emits_traces and not self.case_concepts:
                errors.append(
                    "order_representations includes DF/EF/EF_EXPLICIT but case_concepts is "
                    "empty. Set settings.case_concepts before calling build_ontology()."
                )

            if self.emits_declare and self.declare_result is None:
                errors.append(
                    "order_representations includes DECLARE but no DECLARE constraints are "
                    "available. Set settings.declare_file or assign settings.declare_result."
                )

        if errors:
            raise ValueError(
                "Invalid OntoBuildSettings:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def to_dict(self) -> dict:
        """Return the settings as plain JSON-compatible values.

        This is what a build records as its provenance, so every field that
        changes the output appears here.

        Every field is recorded, so a field added to the settings appears here
        without being listed a second time.

        Returns:
            A dict of primitives, lists and dicts.
        """
        return {
            field.name: _plain(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }


def _plain(value: Any) -> Any:
    """Render a setting as the JSON-compatible value the record holds."""
    if isinstance(value, BuildEnum):
        return value.value
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, DeclareResult):
        return value.to_dict()
    return value
