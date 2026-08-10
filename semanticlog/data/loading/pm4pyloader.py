import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pm4py as pm
from pm4py.objects.ocel.obj import OCEL

from semanticlog.data.loading.abstractloader import AbstractLoader
from semanticlog.data.oced import EntityKind, OCEDStore

if TYPE_CHECKING:
    from semanticlog.onto.settings import OcelExtension

_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".xml": "xml",
    ".xmlocel": "xml",
    ".json": "json",
    ".jsonocel": "json",
    ".sqlite": "sqlite",
}

# Derived so that callers selecting input files cannot drift from what this
# loader actually accepts.
OCEL_SUFFIXES: frozenset[str] = frozenset(_FORMAT_BY_SUFFIX)

# Reader *names* rather than the functions themselves, so that the lookup on the
# pm4py module happens at call time and the readers stay patchable in tests.
_READER_NAMES: dict[str, str] = {
    "xml": "read_ocel2_xml",
    "json": "read_ocel2_json",
    "sqlite": "read_ocel2_sqlite",
}

# Qualifier values pm4py may emit for an unqualified relation.  Both the true
# nulls and the string placeholders are replaced, so both must be counted when
# reporting how many rows were affected.
_MISSING_QUALIFIER_STRINGS: list[str] = ["None", "none", "NA", "N/A", ""]
_MISSING_QUALIFIERS: list = [None, np.nan, *_MISSING_QUALIFIER_STRINGS]
_FALLBACK_QUALIFIER = "modifies"


def _prefix_ids(ids: pd.Series, prefix: str) -> pd.Series:
    """Prefix purely numeric identifiers with their type or activity name."""
    return prefix + "_" + ids.astype(str)


def _normalise_qualifiers(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Return a copy of ``df`` with missing qualifiers replaced by the fallback name.

    Args:
        df: Relation table containing an ``ocel:qualifier`` column.
        kind: ``"e2o"`` or ``"o2o"``, used only in the warning message.

    Returns:
        A copy of ``df``; the caller's DataFrame is never mutated.
    """
    df = df.copy()
    qualifier = df["ocel:qualifier"]
    n_missing = int(
        qualifier.isna().sum() + qualifier.isin(_MISSING_QUALIFIER_STRINGS).sum()
    )
    if n_missing:
        logging.warning(
            f"pm4pyloader: {n_missing} {kind} relation(s) had no qualifier and were "
            f"assigned the fallback relation name '{_FALLBACK_QUALIFIER}'."
        )
    df["ocel:qualifier"] = qualifier.replace(_MISSING_QUALIFIERS, _FALLBACK_QUALIFIER)
    return df


class PM4PYLoader(AbstractLoader):
    """AbstractLoader implementation which uses PM4PY to load OCEL logs and store them in OCEDStore objects."""

    def __init__(
        self,
        path: str,
        config: dict[str, str],
        ocel_extensions: "list[OcelExtension] | None" = None,
    ):
        """Initialize the OCEL data loader with a configuration dict and path.

        Args:
            path: Path to the OCEL 2.0 file (.xml, .xmlocel, .json, .jsonocel, .sqlite).
            config: Configuration dict. Accepts ``file_format`` key to override
                auto-detection from the file extension.
            ocel_extensions: Optional list of ``OcelExtension`` values controlling
                which OCEL 2.0 features are loaded. Pass
                ``settings.ocel_extensions`` to keep the loader and builder in sync.
                Defaults to no extensions (legacy behaviour).
        """
        super().__init__(path, config)
        self._ocel_extensions: list[OcelExtension] = ocel_extensions or []
        self.file_format = self._resolve_file_format(path, config)
        config["file_format"] = self.file_format

    @staticmethod
    def _resolve_file_format(path: str, config: dict[str, str]) -> str:
        given = config.get("file_format")
        if given is not None:
            if given not in _READER_NAMES:
                raise ValueError(
                    f"Unsupported file_format '{given}'. "
                    f"Supported formats: {', '.join(sorted(_READER_NAMES))}."
                )
            return given
        inferred = _FORMAT_BY_SUFFIX.get(Path(path).suffix.lower())
        if inferred is None:
            raise ValueError(
                f"Cannot infer OCEL format from '{path}'. Supported extensions: "
                f"{', '.join(sorted(_FORMAT_BY_SUFFIX))}."
            )
        return inferred

    def get_oced(self) -> OCEDStore:
        """Returns the data as a OCEDStore object"""
        from semanticlog.onto.settings import OcelExtension

        load_attr_history = OcelExtension.ATTRIBUTE_HISTORY in self._ocel_extensions

        logging.info(f"Loading ocel data from file {self.path}..")
        read: Callable[[str], OCEL] = getattr(pm, _READER_NAMES[self.file_format])
        log = read(self.path)

        oced = OCEDStore(ocel_mode=load_attr_history)

        # Objects
        object_dfs = {
            object_type: df for object_type, df in log.objects.groupby("ocel:type")
        }
        reindex_objects = []
        for object_type in object_dfs:
            object_dfs[object_type] = object_dfs[object_type].dropna(axis=1, how="all")
            df = object_dfs[object_type]
            df = df.drop(["ocel:type"], axis=1)
            is_numeric_str = (
                pd.to_numeric(df["ocel:oid"], errors="coerce").notna().all()
            )
            if is_numeric_str:
                reindex_objects.append(object_type)
            if oced.ids_not_used(df["ocel:oid"]):
                oced.add_object(object_type, "ocel:oid", df)
            else:
                raise ValueError("IDs are not unique!")

        # O2O Relations
        o2o = _normalise_qualifiers(log.o2o, "o2o")
        o2o_relation_dfs = {
            relation_type: df for relation_type, df in o2o.groupby("ocel:qualifier")
        }
        for relation_type in o2o_relation_dfs:
            df = o2o_relation_dfs[relation_type]
            relation_type = relation_type + "_o2o"

            # Object types only. Both ends of an o2o relation are objects, and
            # asking for objects explicitly removes the reliance on events not
            # having been loaded yet.
            type_1 = oced.get_types(df["ocel:oid"], of=EntityKind.OBJECT)
            type_2 = oced.get_types(df["ocel:oid_2"], of=EntityKind.OBJECT)

            if not type_1 or not type_2:
                raise ValueError(
                    f"Could not resolve object types for o2o relation '{relation_type}'."
                )

            oced.add_object_to_object_relation(
                relation_type, "ocel:oid", type_1, "ocel:oid_2", type_2, df
            )

        # E2O Relations
        relations = _normalise_qualifiers(log.relations, "e2o")
        e2o_relation_dfs = {
            relation_type: df
            for relation_type, df in relations.groupby("ocel:qualifier")
        }
        for relation_type in e2o_relation_dfs:
            df = e2o_relation_dfs[relation_type]
            object_types = e2o_relation_dfs[relation_type]["ocel:type"].unique()
            e2o_cols = ["ocel:eid", "ocel:oid"]
            oced.add_event_to_object_relation(
                relation_type,
                "ocel:eid",
                "ocel:oid",
                object_types,
                df[e2o_cols],
            )

        # Prefix purely numerical object IDs with their type name to avoid
        # collisions with numerically indexed events.
        for object_type in reindex_objects:
            oced.change_ids(
                EntityKind.OBJECT,
                object_type,
                partial(_prefix_ids, prefix=object_type),
            )

        # Load object attribute change history when ATTRIBUTE_HISTORY is active.
        # Done after numeric-ID renaming so that ocel:oid values in object_changes
        # match the prefixed IDs now stored in the OCEDStore.
        if load_attr_history:
            oc = log.object_changes
            if oc is not None and len(oc) > 0:
                # Apply the same numeric-ID prefix used for objects above
                oc = oc.copy()
                for object_type in reindex_objects:
                    mask = oc["ocel:type"] == object_type
                    oc.loc[mask, "ocel:oid"] = _prefix_ids(
                        oc.loc[mask, "ocel:oid"], object_type
                    )
                for obj_type, grp in oc.groupby("ocel:type"):
                    oced.add_object_changes(str(obj_type), grp.reset_index(drop=True))

        # Events
        event_dfs = {
            activity: df for activity, df in log.events.groupby("ocel:activity")
        }
        reindex_events = []

        ids = []
        for activity in event_dfs:
            ids.extend(event_dfs[activity]["ocel:eid"].unique())

        if len(ids) != len(set(ids)):
            raise ValueError("Event IDs are not unique across activities.")

        for activity in event_dfs:
            df = event_dfs[activity]
            df = df.drop(["ocel:activity"], axis=1)
            is_numeric_str = (
                pd.to_numeric(df["ocel:eid"], errors="coerce").notna().all()
            )
            if is_numeric_str:
                reindex_events.append(activity)
            if oced.ids_not_used(df["ocel:eid"]):
                oced.add_event(activity, "ocel:eid", "ocel:timestamp", df)
            else:
                raise ValueError("IDs are not unique!")

        # Prefix purely numerical event IDs with their activity name.
        for activity in reindex_events:
            oced.change_ids(
                EntityKind.EVENT, activity, partial(_prefix_ids, prefix=activity)
            )

        return oced
