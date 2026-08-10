"""Every supported build configuration either builds or is rejected by ``validate()``.

The matrices here exist because a configuration can fail in three distinct ways
and only the first was previously covered:

1. ``validate()`` rejects it, which is a supported outcome and must stay a
   ``ValueError`` with a readable message.
2. Axiom construction raises from inside owlapy, which is always a semlog bug.
3. It builds, saves, and the reasoner then refuses the file. Only
   :func:`test_owl2dl_order_is_accepted_by_the_reasoner` can see this one, and
   it is what an OWL 2 DL profile claim actually means.
"""

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.data.oced import OCEDStore
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OntoBuildSettings,
    OwlProfile,
)

NS = DEFAULT_IRI

DF = EventOrderRepresentation.DF
EF = EventOrderRepresentation.EF
EF_X = EventOrderRepresentation.EF_explicit

AXIOMATIC = InverseRepresentation.AXIOMATIC
IMPLICIT = InverseRepresentation.IMPLICIT

VALUES = DataRepresentation.VALUES
EQUAL_BINS = DataRepresentation.EQUAL_BINS
DIST_BINS = DataRepresentation.DIST_BINS


@pytest.fixture
def store() -> OCEDStore:
    """One trace of three events over one case concept, with a numeric attribute."""
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
    return data


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
    return {e.iri.get_remainder() for e in entities}


ORDER_MATRIX = [
    pytest.param(
        profile,
        order,
        id=f"{profile.value}-{'+'.join(o.name for o in order) or 'none'}",
    )
    for profile in OwlProfile
    for order in ([], [DF], [EF], [DF, EF], [EF_X], [DF, EF_X], [DF, EF, EF_X])
]


@pytest.mark.parametrize(("profile", "order"), ORDER_MATRIX)
def test_order_representation_matrix(store, profile, order):
    """Every profile/order pair builds, or is rejected with a readable ValueError."""
    # EF drives TBox transitivity axioms, which OWL2EL does not permit.
    el_rejects_ef = profile == OwlProfile.OWL2EL and EF in order

    if el_rejects_ef:
        with pytest.raises(ValueError, match="EF"):
            _build(store, profile=profile, order_representation=order)
        return

    builder = _build(store, profile=profile, order_representation=order)
    names = _names(builder)

    assert {"Event", "Object", "Time"} <= names
    if DF in order:
        assert "df_order" in names
    if EF in order or EF_X in order:
        assert "ef_order" in names
    if not order:
        assert "df_order" not in names


INVERSE_MATRIX = [
    pytest.param(
        profile,
        inverses,
        id=f"{profile.value}-{'+'.join(i.name for i in inverses) or 'none'}",
    )
    for profile in OwlProfile
    for inverses in ([], [AXIOMATIC], [IMPLICIT], [AXIOMATIC, IMPLICIT])
]


@pytest.mark.parametrize(("profile", "inverses"), INVERSE_MATRIX)
def test_inverse_representation_matrix(store, profile, inverses):
    el_rejects_axiomatic = profile == OwlProfile.OWL2EL and AXIOMATIC in inverses

    if el_rejects_axiomatic:
        with pytest.raises(ValueError, match="AXIOMATIC"):
            _build(
                store,
                profile=profile,
                order_representation=[DF],
                inverse_representation=inverses,
            )
        return

    builder = _build(
        store,
        profile=profile,
        order_representation=[DF],
        inverse_representation=inverses,
    )
    names = _names(builder)

    assert ("inv_event_object_relation" in names) == bool(inverses)


DATA_MATRIX = [
    pytest.param(
        profile, reps, id=f"{profile.value}-{'+'.join(r.name for r in reps) or 'none'}"
    )
    for profile in OwlProfile
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
                order_representation=[DF],
                data_representation=reps,
            )
        return

    builder = _build(
        store,
        profile=profile,
        order_representation=[DF],
        data_representation=reps,
    )
    names = _names(builder)

    # The numeric attribute is dropped entirely when no representation is active.
    assert ("price" in names) == bool(reps)


@pytest.mark.parametrize(
    "order",
    [[DF], [EF], [DF, EF], [EF_X], [DF, EF_X], [DF, EF, EF_X]],
    ids=lambda o: "+".join(x.name for x in o),
)
def test_owl2dl_order_is_accepted_by_the_reasoner(store, tmp_path, order):
    """An OWL2DL build must be loadable and consistent, not merely serialisable.

    Guards the OWL 2 DL global restriction on non-simple properties: a property
    cannot be transitive and appear in a self-restriction at the same time. A
    build that violates it saves without complaint and is then rejected on load.
    """
    builder = _build(store, profile=OwlProfile.OWL2DL, order_representation=order)

    path = tmp_path / "matrix.owl"
    builder.save_onto(str(path))

    assert Reasoner(str(path), profile=OwlProfile.OWL2DL).check()
