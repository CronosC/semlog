"""``has_{attr}`` properties carry typing, and shared attribute names are reported.

An attribute class is named after the column alone, so a column on two types
becomes one class. Renaming would change IRIs, so the build reports the overlap
instead of acting on it, and writes a domain covering every owner.
"""

import logging
from datetime import datetime

import pandas as pd
import pytest

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.attributes import attribute_owners, report_conflated_attributes
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.settings import DEFAULT_IRI, OntoBuildSettings

NS = DEFAULT_IRI


def _store(shared: bool) -> OCEDStore:
    """One store where ``name`` is unique to ``order``, one where ``item`` shares it."""
    data = OCEDStore()
    data.add_event(
        "act",
        id_col="eid",
        time_col="t",
        data=pd.DataFrame({"eid": ["e1"], "t": [datetime(2024, 1, 1)]}),
    )
    data.add_object(
        "order", id_col="oid", data=pd.DataFrame({"oid": ["o1"], "name": ["a"]})
    )
    item = {"iid": ["i1"]}
    if shared:
        item["name"] = ["b"]
    data.add_object("item", id_col="iid", data=pd.DataFrame(item))
    data.add_event_to_object_relation(
        "rel",
        from_id_col="eid",
        to_id_col="oid",
        to_type=["order"],
        data=pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )
    return clean_oced_data(data)


def _domains_of(builder: OCEDOntoBuilder, prop: str) -> set[frozenset[str]]:
    """Return one entry per domain axiom, each the class names it mentions.

    Kept per axiom rather than flattened: owlready2 also reports the domain
    inherited from ``has_attribute``, and the point of the union is that the
    owners arrive as a *single* axiom. Merging them would hide exactly that.
    """
    from owlapy.owl_property import OWLObjectProperty

    domains = set()
    for axiom in builder.onto.object_property_domain_axioms(
        OWLObjectProperty(NS + prop)
    ):
        expression = axiom.get_domain()
        operands = getattr(expression, "operands", None)
        parts = list(operands()) if callable(operands) else [expression]
        domains.add(frozenset(p.iri.get_remainder() for p in parts))
    return domains


def test_has_attribute_property_gets_a_range():
    """Without a range the ontology carries no attribute typing at all.

    ``Attribute`` also appears, inherited from the ``has_attribute`` parent.
    """
    from owlapy.owl_property import OWLObjectProperty

    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(_store(shared=False))

    ranges = {
        axiom.get_range().iri.get_remainder()
        for axiom in builder.onto.object_property_range_axioms(
            OWLObjectProperty(NS + "has_name")
        )
    }
    assert "name" in ranges


def test_domain_is_the_single_owner_when_unshared():
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(_store(shared=False))

    assert frozenset({"order"}) in _domains_of(builder, "has_name")


def test_domain_is_one_union_axiom_over_every_owner_when_shared():
    """Two separate domain axioms would be read as an intersection, not a union."""
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(_store(shared=True))

    domains = _domains_of(builder, "has_name")

    assert frozenset({"order", "item"}) in domains
    # Not one axiom per owning table.
    assert frozenset({"order"}) not in domains
    assert frozenset({"item"}) not in domains


def test_attribute_owners_maps_columns_to_every_carrying_type():
    owners = attribute_owners(_store(shared=True))
    assert owners["name"] == ["item", "order"]


def test_conflated_attributes_are_reported(caplog):
    owners = attribute_owners(_store(shared=True))

    with caplog.at_level(logging.WARNING):
        conflated = report_conflated_attributes(owners)

    assert conflated == {"name": ["item", "order"]}
    assert "ATTRIBUTE CONFLATION" in caplog.text
    assert "'name' on item, order" in caplog.text


def test_unshared_attributes_are_not_reported(caplog):
    owners = attribute_owners(_store(shared=False))

    with caplog.at_level(logging.WARNING):
        assert report_conflated_attributes(owners) == {}

    assert "ATTRIBUTE CONFLATION" not in caplog.text


@pytest.mark.parametrize("shared", [False, True])
def test_typed_attributes_stay_consistent(shared, tmp_path):
    """Adding domain and range must not make the ontology unsatisfiable."""
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    semlog = builder.build_ontology(_store(shared=shared))

    assert semlog.check() is True
