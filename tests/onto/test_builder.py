"""The builder's lifecycle, and the sink API a build drives it through.

A builder produces one ontology and then refuses to be used again, and the
settings it was given cannot change under it. The rest is the convenience
layer on `AxiomSink`: what it accepts, what it refuses and what bookkeeping it
keeps while a build runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from semanticlog.core.enums import (
    DataRepresentation,
    EventOrderRepresentation,
    OWLProfile,
)
from semanticlog.core.vocabulary import DEFAULT_IRI, NameCollisionError
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import InMemorySink

NS = DEFAULT_IRI

TEST_IRI = "https://test#"


@pytest.fixture
def plain() -> OntoBuildSettings:
    """Settings that emit the entities and nothing derived from them."""
    return OntoBuildSettings(order_representations=[], case_concepts=[])


@pytest.fixture
def sink(plain) -> InMemorySink:
    return InMemorySink(TEST_IRI, plain)


# --------------------------------------------------------------------------
# One builder, one ontology
# --------------------------------------------------------------------------


def test_a_second_build_is_refused(tiny_store, pipeline_store):
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(tiny_store)

    with pytest.raises(ValueError, match="already produced an ontology"):
        builder.build_ontology(pipeline_store)


def test_a_refused_second_build_does_not_merge_the_two_logs(tiny_store, pipeline_store):
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(tiny_store)

    with pytest.raises(ValueError):
        builder.build_ontology(pipeline_store)

    names = {c.iri.remainder for c in builder.sink.onto.classes_in_signature()}
    assert "step" in names
    assert "place_order" not in names


def test_a_build_rejected_on_its_input_leaves_a_fresh_builder_usable(tiny_store, store):
    settings = OntoBuildSettings(
        case_concepts=["order"],
        order_representations=[EventOrderRepresentation.DECLARE],
    )

    with pytest.raises(ValueError, match="no DECLARE constraints are available"):
        OCEDOntoBuilder(NS, settings).build_ontology(tiny_store)

    usable = OCEDOntoBuilder(NS, OntoBuildSettings())
    assert list(usable.build_ontology(store("tiny")).onto.classes_in_signature())


def test_a_step_with_nothing_to_do_says_so_instead_of_counting_to_zero(
    tiny_store, caplog
):
    builder = OCEDOntoBuilder(NS, OntoBuildSettings(order_representations=[]))

    with caplog.at_level(logging.INFO, logger="semanticlog.onto.builder"):
        builder.build_ontology(tiny_store)

    assert "No object to object relations in the data.." in caplog.text
    assert "No event order to instantiate.." in caplog.text
    assert "(0 relation types)" not in caplog.text


def test_a_step_with_work_to_do_announces_how_much(store, caplog):
    builder = OCEDOntoBuilder(NS, OntoBuildSettings(order_representations=[]))

    with caplog.at_level(logging.INFO, logger="semanticlog.onto.builder"):
        builder.build_ontology(store("pipeline"))

    assert "Instantiating object to object relation data (1 relation types).." in (
        caplog.text
    )
    assert "No object to object relations" not in caplog.text


def test_the_unions_a_profile_refuses_are_reported_once_for_the_build(
    awkward_store, caplog
):
    builder = OCEDOntoBuilder(NS, OntoBuildSettings(profile=OWLProfile.OWL2EL))

    with caplog.at_level(logging.INFO, logger="semanticlog.onto.builder"):
        builder.build_ontology(awkward_store)

    omitted = [r.message for r in caplog.records if "omitted" in r.message]
    assert len(omitted) == 1
    assert "under OWL 2 EL, which has no disjunction" in omitted[0]
    unions, classes = builder.sink.dropped_unions
    assert f"omitted {unions} unions of {classes} classes" in omitted[0]


def test_no_union_is_reported_when_the_profile_keeps_them_all(awkward_store, caplog):
    builder = OCEDOntoBuilder(NS, OntoBuildSettings(profile=OWLProfile.OWL2DL))

    with caplog.at_level(logging.INFO, logger="semanticlog.onto.builder"):
        builder.build_ontology(awkward_store)

    assert builder.sink.dropped_unions == (0, 0)
    assert not [r for r in caplog.records if "omitted" in r.message]


def test_a_build_rejected_on_a_name_spends_the_builder(tiny_store):
    builder = OCEDOntoBuilder(NS, OntoBuildSettings(profile=OWLProfile.OWL2EL))

    dirty = OCEDStore()
    dirty.add_event(
        "bad name",
        id_col="eid",
        time_col="t",
        data=pd.DataFrame({"eid": ["e1"], "t": [pd.Timestamp("2024-01-01")]}),
    )
    with pytest.raises(ValueError, match="cannot be an IRI local name"):
        builder.build_ontology(dirty)

    with pytest.raises(ValueError, match="already produced an ontology"):
        builder.build_ontology(tiny_store)


# --------------------------------------------------------------------------
# The settings a build was given
# --------------------------------------------------------------------------


def test_settings_cannot_be_changed_after_a_build(tiny_store):
    semlog = OCEDOntoBuilder(
        NS, OntoBuildSettings(case_concepts=["order"])
    ).build_ontology(tiny_store)

    with pytest.raises(FrozenInstanceError):
        semlog.settings.derive_annotations = True
    with pytest.raises(AttributeError):
        semlog.settings.case_concepts.append("sneaky")

    assert semlog.settings.derive_annotations is False
    assert semlog.settings.case_concepts == ("order",)


def test_replace_leaves_the_original_alone():
    settings = OntoBuildSettings(case_concepts=["order"])

    changed = settings.replace(profile=OWLProfile.OWL2EL, case_concepts=["item"])

    assert settings.case_concepts == ("order",)
    assert settings.profile is OWLProfile.OWL2DL
    assert changed.case_concepts == ("item",)


def test_a_sequence_passed_in_is_not_shared_with_the_settings():
    concepts = ["order"]
    settings = OntoBuildSettings(case_concepts=concepts)

    concepts.append("item")

    assert settings.case_concepts == ("order",)


def test_a_file_backed_setting_is_read_once_at_construction(tmp_path):
    document = tmp_path / "annotations.json"
    document.write_text(json.dumps({"Event": {"label": "Occurrence"}}))
    settings = OntoBuildSettings(annotation_document=str(document))
    document.unlink()

    assert settings.to_dict()["annotation_data"] == {"Event": {"label": "Occurrence"}}


def test_a_missing_annotation_document_fails_at_construction(tmp_path):
    with pytest.raises(FileNotFoundError):
        OntoBuildSettings(annotation_document=str(tmp_path / "absent.json"))


def test_an_enum_setting_written_as_text_becomes_the_member():
    """The enums subclass str, so a bare string would work until .value."""
    settings = OntoBuildSettings(profile="OWL2EL", data_representations=["equal_bins"])

    assert settings.profile is OWLProfile.OWL2EL
    assert settings.data_representations == (DataRepresentation.EQUAL_BINS,)
    assert settings.to_dict()["profile"] == "OWL2EL"


def test_an_enum_setting_may_also_be_written_as_the_member_name():
    settings = OntoBuildSettings(data_representations=["VALUES"])

    assert settings.data_representations == (DataRepresentation.VALUES,)


def test_an_unknown_enum_setting_is_refused_at_construction():
    with pytest.raises(ValueError, match="profile: unknown value 'OWL2XL'"):
        OntoBuildSettings(profile="OWL2XL")
    with pytest.raises(ValueError, match="data_representations"):
        OntoBuildSettings(data_representations=["bins"])


# --------------------------------------------------------------------------
# The sink API a build writes through
# --------------------------------------------------------------------------


def test_a_class_is_declared_once_however_many_parents_it_gets(sink):
    sink.add_class("Human")
    sink.add_class("Parent", "Human")
    sink.add_class("Male")
    sink.add_class("Man", ["Human"])
    sink.add_class("Man", ["Male"])

    assert len(list(sink.onto.classes_in_signature())) == 4


def test_classes_can_be_declared_in_one_call(sink):
    sink.add_classes(["Human", "Male", "Female"])
    sink.add_classes(["Man", "Women"], ["Human", "Human"])
    sink.add_classes(["Man", "Women"], ["Male", "Female"])
    sink.add_classes(["Child"], [None])

    assert len(list(sink.onto.classes_in_signature())) == 6


def test_individuals_are_asserted_into_their_class(sink):
    sink.add_class("Letter")

    sink.assert_individuals(["a", "b", "c", "d", "e"], "Letter")

    assert len(list(sink.onto.classes_in_signature())) == 1
    assert len(list(sink.onto.individuals_in_signature())) == 5


def test_a_property_can_be_declared_with_a_domain_range_and_inverse(sink):
    sink.add_classes(["Parent", "Child"])
    sink.add_class("Boy", superclass="Child")

    sink.add_property(
        "hasChild",
        domain_class="Parent",
        range_class="Child",
        create_inverse=True,
        inverse_name="hasParent",
    )
    sink.add_property(
        "hasMaleChild", domain_class="Parent", range_class="Boy", create_inverse=True
    )
    sink.add_property(
        "hasAge", domain_class="Child", range_class="integer", data_prop=True
    )

    assert len(list(sink.onto.properties_in_signature())) == 5
    assert len(list(sink.onto.data_properties_in_signature())) == 1


def test_two_properties_cannot_share_one_inverse_name(sink):
    sink.add_classes(["Parent", "Child"])
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


def test_a_relation_and_a_literal_reach_the_written_file(sink, tmp_path):
    path = tmp_path / "written.owl"
    sink.add_class("Person")
    sink.assert_individuals(["Alice", "Bob"], "Person")
    sink.add_property("knows", domain_class="Person", range_class="Person")
    sink.assert_relations(
        pd.DataFrame({"from": ["Alice"], "to": ["Bob"]}),
        prop="knows",
        from_key="from",
        to_key="to",
    )
    sink.add_property(
        "hasAge", domain_class="Person", range_class="integer", data_prop=True
    )
    sink.assert_relations(
        pd.DataFrame({"from": ["Alice"], "to": [25]}),
        prop="hasAge",
        from_key="from",
        to_key="to",
        data_prop=True,
    )
    sink.save(str(path))

    written = path.read_text(encoding="utf-8")
    assert "knows" in written
    assert ">25<" in written


def test_a_datetime_column_is_asserted_as_a_typed_literal(sink, tmp_path):
    path = tmp_path / "written.owl"
    sink.add_class("Event")
    sink.assert_individuals(["e1", "e2"], "Event")
    sink.add_property("occurredAt", domain_class="Event", data_prop=True)

    sink.assert_relations(
        pd.DataFrame(
            {
                "from": ["e1", "e2"],
                "to": pd.date_range("2022-01-01", periods=2, freq="D"),
            }
        ),
        prop="occurredAt",
        from_key="from",
        to_key="to",
        data_prop=True,
    )
    sink.save(str(path))

    written = Path(path).read_text(encoding="utf-8")
    assert "dateTime" in written
    assert "2022-01-01" in written


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("an inverse name without create_inverse", {"inverse_name": "p_inv"}),
        (
            "an inverse on a data property",
            {"range_class": "integer", "data_prop": True, "create_inverse": True},
        ),
        ("a datatype range on an object property", {"range_class": "integer"}),
        (
            "an unknown datatype",
            {"range_class": "stringy", "data_prop": True},
        ),
    ],
)
def test_a_property_declared_inconsistently_is_refused(sink, name, kwargs):
    sink.add_class("A")

    with pytest.raises(ValueError):
        sink.add_property("p", domain_class="A", **kwargs)
