"""Mined DECLARE constraints, kept apart per case concept, with the settings they were mined with.

`DeclareResult` is written to and read from one JSON file.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from semanticlog.core.errors import InputError

# The templates Declare4Py parses, from its DeclareModel grammar. Unary ones
# take one activity, the rest take an ordered pair.
UNARY_TEMPLATES: tuple[str, ...] = (
    "Absence1",
    "Absence2",
    "Absence3",
    "Existence1",
    "Existence2",
    "Existence3",
    "Exactly1",
    "Exactly2",
    "End",
    "Init",
)

BINARY_TEMPLATES: tuple[str, ...] = (
    "Choice",
    "Exclusive Choice",
    "Responded Existence",
    "Response",
    "Alternate Response",
    "Chain Response",
    "Precedence",
    "Alternate Precedence",
    "Chain Precedence",
    "Succession",
    "Alternate Succession",
    "Chain Succession",
    "Co-Existence",
    "Not Co-Existence",
    "Not Succession",
    "Not Chain Succession",
)

TEMPLATES: frozenset[str] = frozenset(UNARY_TEMPLATES) | frozenset(BINARY_TEMPLATES)

_BY_COMPACT_NAME: dict[str, str] = {t.replace(" ", ""): t for t in TEMPLATES}


def template_name(compact: str) -> str:
    """Return a template as Declare4Py spells it, from the form without spaces a constraint name holds.

    A name that is no known template is returned as it is.
    """
    return _BY_COMPACT_NAME.get(compact, compact)


@dataclass(frozen=True)
class MiningSettings:
    """How a `DeclareResult` was mined.

    Attributes:
        method: `discovery` or `enumeration`.
        templates: The templates kept, or None for every template.
        constraint_support: Share of traces a discovered constraint had to hold in. None under enumeration.
        itemset_support: Share of traces an activity set had to occur in. None under enumeration.
    """

    method: str
    templates: tuple[str, ...] | None
    constraint_support: float | None = None
    itemset_support: float | None = None

    def __post_init__(self) -> None:
        if self.templates is not None:
            object.__setattr__(self, "templates", tuple(self.templates))


@dataclass(frozen=True)
class ConceptConstraints:
    """The constraints checked over the traces of one case concept.

    Attributes:
        case_concept: Object type whose instances are the traces.
        trace_ids: Every trace the constraints were checked against.
        satisfied: Constraint class name to the traces that satisfy it.

    Raises:
        InputError: If a trace is listed twice, or a satisfying trace is not among `trace_ids`.
    """

    case_concept: str
    trace_ids: tuple[str, ...]
    satisfied: Mapping[str, frozenset[str]]

    def __post_init__(self) -> None:
        trace_ids = tuple(map(str, self.trace_ids))
        repeated = sorted(t for t, n in Counter(trace_ids).items() if n > 1)
        if repeated:
            raise InputError(
                f"DECLARE constraints over {self.case_concept!r} list trace(s) "
                f"{', '.join(repeated[:5])} more than once. Mine them again with "
                "'semlog mine-declare'."
            )
        satisfied = {
            name: frozenset(map(str, ids)) for name, ids in self.satisfied.items()
        }
        known = set(trace_ids)
        stray = sorted({t for ids in satisfied.values() for t in ids} - known)
        if stray:
            raise InputError(
                f"DECLARE constraints over {self.case_concept!r} are satisfied by "
                f"trace(s) {', '.join(stray[:5])} that are not among its traces. "
                "Mine them again with 'semlog mine-declare'."
            )
        object.__setattr__(self, "trace_ids", trace_ids)
        object.__setattr__(self, "satisfied", MappingProxyType(satisfied))


@dataclass(frozen=True)
class DeclareResult:
    """DECLARE constraints mined from one log, per case concept.

    A constraint name mined for several case concepts is one constraint. Its members are the satisfying traces of each.

    Attributes:
        concepts: One entry per case concept, in mining order.
        mining: The settings the constraints were mined with.
        source_log: Path of the log they were mined from.

    Raises:
        InputError: If a case concept appears twice.
    """

    concepts: tuple[ConceptConstraints, ...]
    mining: MiningSettings
    source_log: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "concepts", tuple(self.concepts))
        repeated = sorted(n for n, k in Counter(self.case_concepts).items() if k > 1)
        if repeated:
            raise InputError(
                f"DECLARE constraints list case concept(s) "
                f"{', '.join(map(repr, repeated))} more than once. Mine them again "
                "with 'semlog mine-declare'."
            )

    @property
    def case_concepts(self) -> list[str]:
        """The case concepts, in mining order."""
        return [c.case_concept for c in self.concepts]

    def constraint_names(self) -> list[str]:
        """Return every constraint name, in the order the concepts first list them."""
        return list(dict.fromkeys(n for c in self.concepts for n in c.satisfied))

    def case_concepts_of(self, constraint: str) -> list[str]:
        """Return the case concepts with a trace that satisfies `constraint`, in mining order."""
        return [c.case_concept for c in self.concepts if c.satisfied.get(constraint)]

    def members(self, constraint: str) -> list[str]:
        """Return the traces that satisfy `constraint`, concept by concept in trace order."""
        return [
            trace
            for c in self.concepts
            for trace in c.trace_ids
            if trace in c.satisfied.get(constraint, ())
        ]

    def write(self, path: str | Path) -> None:
        """Write the result as JSON."""
        mining = asdict(self.mining)
        if self.mining.templates is not None:
            mining["templates"] = list(self.mining.templates)
        document = {
            "source_log": self.source_log,
            "mining": mining,
            "concepts": {
                c.case_concept: {
                    "trace_ids": list(c.trace_ids),
                    "satisfied": {
                        name: [t for t in c.trace_ids if t in ids]
                        for name, ids in c.satisfied.items()
                    },
                }
                for c in self.concepts
            },
        }
        Path(path).write_text(
            json.dumps(document, separators=(",", ":")), encoding="utf-8"
        )

    @classmethod
    def read(cls, path: str | Path) -> DeclareResult:
        """Read a result `write` wrote.

        Raises:
            InputError: If the file is not JSON or does not have the structure `write` gives it.
        """
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InputError(
                f"{path} is not readable as JSON ({exc}). Mine the constraints "
                "again with 'semlog mine-declare'."
            ) from exc
        try:
            return cls._from_document(document)
        except (KeyError, TypeError, AttributeError) as exc:
            raise InputError(
                f"{path} is not a DECLARE result: {type(exc).__name__} {exc}. "
                "Mine the constraints again with 'semlog mine-declare'."
            ) from exc

    @classmethod
    def _from_document(cls, document: dict[str, Any]) -> DeclareResult:
        concepts = document["concepts"]
        if not isinstance(concepts, dict):
            raise TypeError("'concepts' is not an object")
        return cls(
            concepts=tuple(
                ConceptConstraints(
                    name,
                    tuple(entry["trace_ids"]),
                    {c: frozenset(ids) for c, ids in entry["satisfied"].items()},
                )
                for name, entry in concepts.items()
            ),
            mining=MiningSettings(**document["mining"]),
            source_log=document.get("source_log"),
        )
