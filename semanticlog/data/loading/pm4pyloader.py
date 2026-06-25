import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pm4py as pm

from semanticlog.data.loading.abstractloader import AbstractLoader
from semanticlog.data.oced import OCEDStore

if TYPE_CHECKING:
    from semanticlog.onto.settings import OcelExtension


def _type_prefix(ids: pd.Series, object_type: str) -> pd.Series:
    return object_type + "_" + ids.astype(str)


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
        if "file_format" not in config:
            if path.endswith(".xml") or path.endswith(".xmlocel"):
                config["file_format"] = "xml"
            elif path.endswith(".json") or path.endswith(".jsonocel"):
                config["file_format"] = "json"
            elif path.endswith(".sqlite"):
                config["file_format"] = "sqlite"
            else:
                raise ValueError("Unknown file type.")

    def get_oced(self) -> OCEDStore:
        """Returns the data as a OCEDStore object"""
        from semanticlog.onto.settings import OcelExtension

        load_attr_history = OcelExtension.ATTRIBUTE_HISTORY in self._ocel_extensions

        logging.info(f"Loading ocel data from file {self.path}..")
        if self.config["file_format"] == "xml":
            log = pm.read_ocel2_xml(self.path)
        elif self.config["file_format"] == "json":
            log = pm.read_ocel2_json(self.path)
        elif self.config["file_format"] == "sqlite":
            log = pm.read_ocel2_sqlite(self.path)

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
        n_o2o_null = log.o2o["ocel:qualifier"].isna().sum()
        if n_o2o_null:
            logging.warning(
                f"pm4pyloader: {n_o2o_null} o2o relation(s) had no qualifier "
                "and were assigned the fallback relation name 'modifies'."
            )
        log.o2o["ocel:qualifier"] = log.o2o["ocel:qualifier"].replace(
            [None, np.nan, "None", "none", "NA", "N/A", ""], "modifies"
        )
        o2o_relation_dfs = {
            relation_type: df for relation_type, df in log.o2o.groupby("ocel:qualifier")
        }
        for relation_type in o2o_relation_dfs:
            df = o2o_relation_dfs[relation_type]
            relation_type = relation_type + "_o2o"

            type_1 = oced.get_types(df["ocel:oid"])
            type_2 = oced.get_types(df["ocel:oid_2"])

            if not type_1 or not type_2:
                raise ValueError(
                    f"Could not resolve object types for o2o relation '{relation_type}'."
                )

            oced.add_object_to_object_relation(
                relation_type, "ocel:oid", type_1, "ocel:oid_2", type_2, df
            )

        # E2O Relations
        relations = log.relations
        n_e2o_null = (
            relations["ocel:qualifier"].isna().sum()
            + relations["ocel:qualifier"].isin(["None", "none", "NA", "N/A", ""]).sum()
        )
        if n_e2o_null:
            logging.warning(
                f"pm4pyloader: {n_e2o_null} e2o relation(s) had no qualifier "
                "and were assigned the fallback relation name 'modifies'."
            )
        relations["ocel:qualifier"] = relations["ocel:qualifier"].replace(
            [None, np.nan, "None", "none", "NA", "N/A", ""], "modifies"
        )
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
                "object",
                object_type,
                lambda x, ot=object_type: _type_prefix(x, ot),
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
                    oc.loc[mask, "ocel:oid"] = _type_prefix(
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
                "event", activity, lambda x, a=activity: a + "_" + x.astype(str)
            )

        return oced
