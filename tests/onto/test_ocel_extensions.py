"""Tests for OcelExtension instantiation: ATTRIBUTE_HISTORY and RELATION_TIMESTAMPS."""

import pandas as pd
import pytest
from owlapy.iri import IRI
from owlapy.owl_property import OWLObjectProperty
from owlapy.owl_reasoner import StructuralReasoner

from semanticlog.data.oced import OCEDStore
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    OcelExtension,
    OntoBuildSettings,
    OwlProfile,
)
from semanticlog.onto.upper import OCEDUpperOntoBuilder

NS = DEFAULT_IRI


def _build_upper(settings, path):
    b = OCEDUpperOntoBuilder(NS, settings)
    b.construct_upper_ontology(path)


def _individual_iris(onto) -> set[str]:
    return {i.iri.remainder for i in onto.individuals_in_signature()}


def _object_prop_assertions(onto, prop_name: str) -> set[tuple[str, str]]:
    """Return all (subject, object) pairs for the given object property."""
    prop = OWLObjectProperty(IRI(NS, prop_name))
    reasoner = StructuralReasoner(onto)
    pairs = set()
    for ind in onto.individuals_in_signature():
        for val in reasoner.object_property_values(ind, prop):
            pairs.add((ind.iri.remainder, val.iri.remainder))
    return pairs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings_ah():
    s = OntoBuildSettings(ocel_extensions=[OcelExtension.ATTRIBUTE_HISTORY])
    s.profile = OwlProfile.OWL2EL
    s.order_representation = []
    s.case_concepts = []
    s.inverse_properties = False
    s.make_inverses_explicit = False
    return s


@pytest.fixture
def settings_rt():
    s = OntoBuildSettings(ocel_extensions=[OcelExtension.RELATION_TIMESTAMPS])
    s.profile = OwlProfile.OWL2EL
    s.order_representation = []
    s.case_concepts = []
    s.inverse_properties = False
    s.make_inverses_explicit = False
    return s


@pytest.fixture
def oced_with_changes():
    """OCEDStore in ocel_mode with one object type that has attribute history."""
    oced = OCEDStore(ocel_mode=True)
    oced.add_object(
        "order",
        "ocel:oid",
        pd.DataFrame({"ocel:oid": ["o1", "o2"]}),
    )
    ev = pd.DataFrame(
        {
            "ocel:eid": ["e1"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01"], utc=True),
        }
    )
    oced.add_event("place_order", "ocel:eid", "ocel:timestamp", ev)
    oced.add_event_to_object_relation(
        "modifies",
        "ocel:eid",
        "ocel:oid",
        ["order"],
        pd.DataFrame({"ocel:eid": ["e1"], "ocel:oid": ["o1"]}),
    )
    changes = pd.DataFrame(
        {
            "ocel:oid": ["o1", "o1", "o2"],
            "ocel:type": ["order", "order", "order"],
            "ocel:timestamp": pd.to_datetime(
                ["2022-01-01", "2022-01-03", "2022-01-02"], utc=True
            ),
            "ocel:field": ["status", "status", "status"],
            "status": ["pending", "shipped", "pending"],
        }
    )
    oced.add_object_changes("order", changes)
    return oced


@pytest.fixture
def oced_with_rel_timestamps():
    """OCEDStore with an e2o relation that carries ocel:timestamp."""
    oced = OCEDStore()
    oced.add_object(
        "order",
        "ocel:oid",
        pd.DataFrame({"ocel:oid": ["o1", "o2"]}),
    )
    ev = pd.DataFrame(
        {
            "ocel:eid": ["e1", "e2"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01", "2022-01-02"], utc=True),
        }
    )
    oced.add_event("act", "ocel:eid", "ocel:timestamp", ev)
    rel_data = pd.DataFrame(
        {
            "ocel:eid": ["e1", "e2"],
            "ocel:oid": ["o1", "o2"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01", "2022-01-02"], utc=True),
        }
    )
    oced.add_event_to_object_relation(
        "modifies", "ocel:eid", "ocel:oid", ["order"], rel_data
    )
    return oced


# ---------------------------------------------------------------------------
# Tests — ATTRIBUTE_HISTORY ABox instantiation
# ---------------------------------------------------------------------------


def test_attribute_history_creates_attribute_change_individuals(
    tmp_path, settings_ah, oced_with_changes
):
    upper_path = str(tmp_path / "upper.owl")
    _build_upper(settings_ah, upper_path)
    builder = OCEDOntoBuilder(upper_path, settings_ah)
    builder.build_ontology(oced_with_changes)

    iris = _individual_iris(builder.onto)
    attr_changes = {i for i in iris if i.startswith("AttrChange_")}
    # 3 change records → 3 AttrChange individuals
    assert len(attr_changes) == 3


def test_attribute_history_links_object_to_attr_change(
    tmp_path, settings_ah, oced_with_changes
):
    upper_path = str(tmp_path / "upper.owl")
    _build_upper(settings_ah, upper_path)
    builder = OCEDOntoBuilder(upper_path, settings_ah)
    builder.build_ontology(oced_with_changes)

    pairs = _object_prop_assertions(builder.onto, "has_attribute_change")
    subjects = {s for s, _ in pairs}
    assert "o1" in subjects
    assert "o2" in subjects


def test_attribute_history_links_attr_change_to_value(
    tmp_path, settings_ah, oced_with_changes
):
    upper_path = str(tmp_path / "upper.owl")
    _build_upper(settings_ah, upper_path)
    builder = OCEDOntoBuilder(upper_path, settings_ah)
    builder.build_ontology(oced_with_changes)

    pairs = _object_prop_assertions(builder.onto, "attribute_change_value")
    objects = {o for _, o in pairs}
    assert "status_pending" in objects
    assert "status_shipped" in objects


def test_attribute_history_links_attr_change_to_time(
    tmp_path, settings_ah, oced_with_changes
):
    upper_path = str(tmp_path / "upper.owl")
    _build_upper(settings_ah, upper_path)
    builder = OCEDOntoBuilder(upper_path, settings_ah)
    builder.build_ontology(oced_with_changes)

    pairs = _object_prop_assertions(builder.onto, "changed_at")
    assert len(pairs) == 3  # one per change record


def test_attribute_history_skipped_when_store_not_in_ocel_mode(tmp_path, settings_ah):
    """If the store was not created with ocel_mode=True, the builder skips gracefully."""
    oced = OCEDStore()  # ocel_mode=False
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1"]}))
    ev = pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2022-01-01"], utc=True)})
    oced.add_event("act", "eid", "ts", ev)
    oced.add_event_to_object_relation(
        "modifies",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )

    upper_path = str(tmp_path / "upper.owl")
    _build_upper(settings_ah, upper_path)
    builder = OCEDOntoBuilder(upper_path, settings_ah)
    builder.build_ontology(oced)  # must not raise

    iris = _individual_iris(builder.onto)
    assert not any(i.startswith("AttrChange_") for i in iris)


# ---------------------------------------------------------------------------
# Tests — RELATION_TIMESTAMPS ABox instantiation
# ---------------------------------------------------------------------------


def test_relation_timestamps_creates_relation_observation_individuals(
    tmp_path, settings_rt, oced_with_rel_timestamps
):
    upper_path = str(tmp_path / "upper.owl")
    _build_upper(settings_rt, upper_path)
    builder = OCEDOntoBuilder(upper_path, settings_rt)
    builder.build_ontology(oced_with_rel_timestamps)

    iris = _individual_iris(builder.onto)
    rel_obs = {i for i in iris if i.startswith("RelObs_")}
    assert len(rel_obs) == 2


def test_relation_timestamps_links_event_to_rel_obs(
    tmp_path, settings_rt, oced_with_rel_timestamps
):
    upper_path = str(tmp_path / "upper.owl")
    _build_upper(settings_rt, upper_path)
    builder = OCEDOntoBuilder(upper_path, settings_rt)
    builder.build_ontology(oced_with_rel_timestamps)

    pairs = _object_prop_assertions(builder.onto, "has_relation_observation")
    subjects = {s for s, _ in pairs}
    assert "e1" in subjects
    assert "e2" in subjects


def test_relation_timestamps_links_rel_obs_to_object(
    tmp_path, settings_rt, oced_with_rel_timestamps
):
    upper_path = str(tmp_path / "upper.owl")
    _build_upper(settings_rt, upper_path)
    builder = OCEDOntoBuilder(upper_path, settings_rt)
    builder.build_ontology(oced_with_rel_timestamps)

    pairs = _object_prop_assertions(builder.onto, "relation_involves_object")
    objects = {o for _, o in pairs}
    assert "o1" in objects
    assert "o2" in objects


def test_relation_timestamps_links_rel_obs_to_time(
    tmp_path, settings_rt, oced_with_rel_timestamps
):
    upper_path = str(tmp_path / "upper.owl")
    _build_upper(settings_rt, upper_path)
    builder = OCEDOntoBuilder(upper_path, settings_rt)
    builder.build_ontology(oced_with_rel_timestamps)

    pairs = _object_prop_assertions(builder.onto, "relation_observed_at")
    assert len(pairs) == 2  # one per relation row


def test_relation_timestamps_skipped_when_no_timestamp_column(tmp_path, settings_rt):
    """Relations without ocel:timestamp column are silently skipped."""
    oced = OCEDStore()
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1"]}))
    ev = pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2022-01-01"], utc=True)})
    oced.add_event("act", "eid", "ts", ev)
    # No ocel:timestamp column in e2o data
    oced.add_event_to_object_relation(
        "modifies",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )

    upper_path = str(tmp_path / "upper.owl")
    _build_upper(settings_rt, upper_path)
    builder = OCEDOntoBuilder(upper_path, settings_rt)
    builder.build_ontology(oced)  # must not raise

    iris = _individual_iris(builder.onto)
    assert not any(i.startswith("RelObs_") for i in iris)
