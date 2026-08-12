"""The sink calls a build makes, at the level of single axioms."""

from pathlib import Path

import pandas as pd
import pytest

from semanticlog.onto.settings import OntoBuildSettings, OWLProfile
from semanticlog.onto.sinks import InMemorySink


@pytest.fixture
def settings() -> OntoBuildSettings:
    return OntoBuildSettings(
        profile=OWLProfile.OWL2DL,
        order_representations=[],
        case_concepts=[],
    )


@pytest.fixture
def sink(settings) -> InMemorySink:
    return InMemorySink("https://test#", settings)


@pytest.fixture
def out_path(tmp_path) -> str:
    return str(tmp_path / "temp.owl")


def test_add_class(sink):
    assert len(list(sink.onto.classes_in_signature())) == 0
    sink.add_class("Human")
    sink.add_class("Parent", "Human")
    sink.add_class("Male")
    sink.add_class("Man", ["Human"])
    sink.add_class("Man", ["Male"])
    assert len(list(sink.onto.classes_in_signature())) == 4


def test_add_classes(sink):
    sink.add_classes(["Human", "Male", "Female"])
    sink.add_classes(["Man", "Women"], ["Human", "Human"])
    sink.add_classes(["Man", "Women"], ["Male", "Female"])
    sink.add_classes(["Child"], [None])
    assert len(list(sink.onto.classes_in_signature())) == 6


def test_assert_individual(sink, out_path):
    individuals = ["a", "b", "c", "d", "e"]
    sink.add_class("Letter")
    for ind in individuals:
        sink.assert_individual(ind, "Letter")
    assert len(list(sink.onto.classes_in_signature())) == 1
    assert len(list(sink.onto.individuals_in_signature())) == 5
    sink.save(out_path)


def test_assert_individuals(sink, out_path):
    individuals = ["a", "b", "c", "d", "e"]
    sink.add_class("Letter")
    sink.assert_individuals(individuals, "Letter")
    assert len(list(sink.onto.classes_in_signature())) == 1
    assert len(list(sink.onto.individuals_in_signature())) == 5
    sink.save(out_path)


def test_add_property(sink):
    sink.add_property("isRelated")
    sink.add_class("Parent")
    sink.add_class("Child")
    sink.add_class("Boy", superclass="Child")
    assert len(list(sink.onto.classes_in_signature())) == 3
    sink.add_property(
        "hasChild",
        domain_class="Parent",
        range_class="Child",
        create_inverse=True,
        inverse_name="hasParent",
    )
    sink.add_property(
        "hasMaleChild",
        domain_class="Parent",
        range_class="Boy",
        create_inverse=True,
    )
    sink.add_property(
        "hasAge", domain_class="Child", range_class="integer", data_prop=True
    )
    assert len(list(sink.onto.properties_in_signature())) == 6
    assert len(list(sink.onto.data_properties_in_signature())) == 1


def test_two_properties_cannot_share_one_inverse_name(sink):
    from semanticlog.core.vocabulary import NameCollisionError

    sink.add_class("Parent")
    sink.add_class("Child")
    sink.add_property(
        "hasChild",
        domain_class="Parent",
        range_class="Child",
        create_inverse=True,
        inverse_name="hasParent",
    )
    assert sink.inverse_name("hasParent") == "hasChild"

    with pytest.raises(NameCollisionError, match="already the inverse of 'hasChild'"):
        sink.add_property(
            "hasMaleChild",
            domain_class="Parent",
            range_class="Child",
            create_inverse=True,
            inverse_name="hasParent",
        )
    assert sink.inverse_name("hasParent") == "hasChild"


def test_assert_relation(sink, out_path):
    sink.add_class("Person")
    individuals = ["Alice", "Bob"]
    sink.assert_individuals(individuals, "Person")
    sink.add_property(
        "knows",
        domain_class="Person",
        range_class="Person",
        create_inverse=False,
    )
    sink.assert_relation("knows", "Alice", "Bob")
    sink.add_property(
        "hasAge", domain_class="Person", range_class="integer", data_prop=True
    )
    sink.assert_relation("hasAge", "Alice", 25, data_prop=True)
    sink.assert_relation("hasAge", "Bob", 34, data_prop=True)
    sink.save(out_path)

    written = Path(out_path).read_text(encoding="utf-8")
    assert "knows" in written
    assert ">25<" in written
    assert ">34<" in written


def test_assert_data_relations_with_datetimes(sink, out_path):
    sink.add_class("Event")
    sink.assert_individuals(["e1", "e2"], "Event")

    sink.add_property(
        "occurredAt",
        domain_class="Event",
        range_class=None,
        data_prop=True,
    )

    df = pd.DataFrame(
        {
            "from": ["e1", "e2"],
            "to": pd.date_range("2022-01-01", periods=2, freq="D"),
        }
    )

    sink.assert_relations(
        df, prop="occurredAt", from_key="from", to_key="to", data_prop=True
    )
    sink.save(out_path)

    written = Path(out_path).read_text(encoding="utf-8")
    assert "dateTime" in written
    assert "2022-01-01" in written


def test_add_property_validation(sink):
    sink.add_class("A")
    sink.add_class("B")

    with pytest.raises(ValueError):
        sink.add_property(
            "p_bad_inv_name",
            domain_class="A",
            range_class="B",
            create_inverse=False,
            inverse_name="p_bad_inv",
        )

    with pytest.raises(ValueError):
        sink.add_property(
            "age_with_inverse",
            domain_class="A",
            range_class="integer",
            data_prop=True,
            create_inverse=True,
        )

    with pytest.raises(ValueError):
        sink.add_property(
            "p_obj_with_data_range",
            domain_class="A",
            range_class="integer",
            data_prop=False,
        )

    with pytest.raises(ValueError):
        sink.add_property(
            "p_data_with_unknown_range",
            domain_class="A",
            range_class="stringy",
            data_prop=True,
        )

    sink.add_property(
        "relatesTo", domain_class="A", range_class="B", create_inverse=True
    )
    sink.add_property("hasAge", domain_class="A", range_class="integer", data_prop=True)

    obj_props = list(sink.onto.object_properties_in_signature())
    data_props = list(sink.onto.data_properties_in_signature())
    assert len(obj_props) >= 2
    assert len(data_props) >= 1
