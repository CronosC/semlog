"""Every combination of profile, order, inverses and data representation.

Each cell either builds or raises a documented ValueError. Nothing else
is an acceptable outcome.
"""

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OntoBuildSettings,
    OWLProfile,
)

NS = DEFAULT_IRI

DF = EventOrderRepresentation.DF
EF = EventOrderRepresentation.EF
EF_X = EventOrderRepresentation.EF_EXPLICIT

AXIOMATIC = InverseRepresentation.AXIOMATIC
IMPLICIT = InverseRepresentation.IMPLICIT

VALUES = DataRepresentation.VALUES
EQUAL_BINS = DataRepresentation.EQUAL_BINS
DIST_BINS = DataRepresentation.DIST_BINS


@pytest.fixture
def store() -> OCEDStore:
    data = OCEDStore()
    data.add_event(
        "place_order",
        id_col="id",
        time_col="time",
        data=pd.DataFrame(
            {
                "id": ["e1", "e2", "e3"],
                "time": [datetime(2023, 1, d) for d in (1, 2, 3)],
                "price": [10.0, 20.0, 30.0],
            }
        ),
    )
    data.add_object("order", id_col="id", data=pd.DataFrame({"id": ["o1"]}))
    data.add_event_to_object_relation(
        "concerns",
        from_id_col="event_id",
        to_id_col="object_id",
        to_type=["order"],
        data=pd.DataFrame(
            {"event_id": ["e1", "e2", "e3"], "object_id": ["o1", "o1", "o1"]}
        ),
    )
    return clean_oced_data(data)


def _build(store: OCEDStore, **kwargs) -> OCEDOntoBuilder:
    settings = OntoBuildSettings(case_concepts=["order"], **kwargs)
    builder = OCEDOntoBuilder(NS, settings)
    builder.build_ontology(store)
    return builder


def _names(builder: OCEDOntoBuilder) -> set[str]:
    entities = (
        list(builder.onto.classes_in_signature())
        + list(builder.onto.object_properties_in_signature())
        + list(builder.onto.individuals_in_signature())
    )
    return {e.iri.remainder for e in entities}


ORDER_MATRIX = [
    pytest.param(
        profile,
        order,
        id=f"{profile.value}-{'+'.join(o.name for o in order) or 'none'}",
    )
    for profile in OWLProfile
    for order in ([], [DF], [EF], [DF, EF], [EF_X], [DF, EF_X], [DF, EF, EF_X])
]


@pytest.mark.parametrize(("profile", "order"), ORDER_MATRIX)
def test_order_representation_matrix(store, profile, order):
    el_rejects_ef = profile == OWLProfile.OWL2EL and EF in order

    if el_rejects_ef:
        with pytest.raises(ValueError, match="EF"):
            _build(store, profile=profile, order_representations=order)
        return

    builder = _build(store, profile=profile, order_representations=order)
    names = _names(builder)

    assert {"Event", "Object", "Time"} <= names

    assert ("df_order" in names) == bool(order)
    assert ("ef_order" in names) == (EF in order or EF_X in order)
    assert ("StartEvent_order" in names) == bool(order)


INVERSE_MATRIX = [
    pytest.param(
        profile,
        inverses,
        id=f"{profile.value}-{'+'.join(i.name for i in inverses) or 'none'}",
    )
    for profile in OWLProfile
    for inverses in ([], [AXIOMATIC], [IMPLICIT], [AXIOMATIC, IMPLICIT])
]


@pytest.mark.parametrize(("profile", "inverses"), INVERSE_MATRIX)
def test_inverse_representation_matrix(store, profile, inverses):
    el_rejects_axiomatic = profile == OWLProfile.OWL2EL and AXIOMATIC in inverses

    if el_rejects_axiomatic:
        with pytest.raises(ValueError, match="AXIOMATIC"):
            _build(
                store,
                profile=profile,
                order_representations=[DF],
                inverse_representations=inverses,
            )
        return

    builder = _build(
        store,
        profile=profile,
        order_representations=[DF],
        inverse_representations=inverses,
    )
    names = _names(builder)

    assert ("inv_event_object_relation" in names) == bool(inverses)


DATA_MATRIX = [
    pytest.param(
        profile, reps, id=f"{profile.value}-{'+'.join(r.name for r in reps) or 'none'}"
    )
    for profile in OWLProfile
    for reps in (
        [],
        [VALUES],
        [EQUAL_BINS],
        [DIST_BINS],
        [VALUES, EQUAL_BINS],
        [EQUAL_BINS, DIST_BINS],
    )
]


@pytest.mark.parametrize(("profile", "reps"), DATA_MATRIX)
def test_data_representation_matrix(store, profile, reps):
    bins_conflict = EQUAL_BINS in reps and DIST_BINS in reps

    if bins_conflict:
        with pytest.raises(ValueError, match="mutually exclusive"):
            _build(
                store,
                profile=profile,
                order_representations=[DF],
                data_representations=reps,
            )
        return

    builder = _build(
        store,
        profile=profile,
        order_representations=[DF],
        data_representations=reps,
    )
    names = _names(builder)

    assert ("price" in names) == bool(reps)

    individuals = {i.iri.remainder for i in builder.onto.individuals_in_signature()}
    if reps:
        assert any(name.startswith("price_") for name in individuals), (
            "a numeric attribute with an active representation must produce at "
            "least one value individual"
        )


@pytest.mark.parametrize(
    "order",
    [[DF], [EF], [DF, EF], [EF_X], [DF, EF_X], [DF, EF, EF_X]],
    ids=lambda o: "+".join(x.name for x in o),
)
@pytest.mark.reasoner
def test_owl2dl_order_is_accepted_by_the_reasoner(store, tmp_path, order):
    builder = _build(store, profile=OWLProfile.OWL2DL, order_representations=order)

    path = tmp_path / "matrix.owl"
    builder.sink.save(str(path))

    assert Reasoner(str(path), profile=OWLProfile.OWL2DL).check()
