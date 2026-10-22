import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.data.oced import attribute_owners_by_kind
from semanticlog.data.tables import EntityKind
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters.attributes import AttributeEmitter
from semanticlog.onto.emitters.entities import EntityEmitter
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.settings import OntoBuildSettings
from tests.support.build import emit
from tests.support.settings import settings


def _emit(sink, data, kind: EntityKind) -> None:
    settings = OntoBuildSettings()
    parts = (sink, settings, AnnotationWriter(sink, settings, {}), EmittedIndividuals())
    attributes = AttributeEmitter(*parts, attribute_owners_by_kind(data), {})
    EntityEmitter(*parts, attributes).emit(data, kind)


@pytest.mark.parametrize(
    ("kind", "parent", "name", "ids"),
    [
        (EntityKind.EVENT, vocab.EVENT, "step", ["e1", "e2", "e3"]),
        (EntityKind.OBJECT, vocab.OBJECT, "order", ["o1"]),
    ],
)
def test_each_type_becomes_a_class_under_its_kind_holding_its_entities(
    recording_sink, tiny_store, kind, parent, name, ids
):
    sink = recording_sink()
    _emit(sink, tiny_store, kind)

    assert ("sub_class", name, [parent]) in sink.calls
    assert [i for _, i, c in sink.of_kind("individual") if c == name] == ids


def test_every_type_class_is_declared_before_any_entity_is_asserted(
    recording_sink, awkward_store
):
    sink = recording_sink()
    _emit(sink, awkward_store, EntityKind.EVENT)
    first_individual = sink.calls.index(sink.of_kind("individual")[0])
    type_classes = {
        call[1]: i
        for i, call in enumerate(sink.calls)
        if call[0] == "sub_class" and call[2] == [vocab.EVENT]
    }

    assert type_classes.keys() == {"place_order", "ship"}
    assert max(type_classes.values()) < first_individual


def test_the_event_timestamp_is_not_emitted_as_an_attribute(recording_sink, tiny_store):
    sink = recording_sink()
    _emit(sink, tiny_store, EntityKind.EVENT)
    owned = {call[1] for call in sink.of_kind("relation")}

    assert owned == {
        vocab.has_attribute_property("step.channel"),
        vocab.has_attribute_property("step.price"),
    }


def test_the_activity_and_object_type_classes_are_disjoint(tmp_path):
    onto = emit(settings(), tmp_path / "matrix.ofn")

    assert {
        frozenset({"place_order", "ship"}),
        frozenset({"order", "item"}),
    } <= onto.disjoint


@pytest.mark.reasoner
def test_an_object_of_two_object_types_is_inconsistent(tmp_path):
    path = tmp_path / "matrix.ofn"
    emit(settings(), path)
    assert Reasoner(str(path)).check() is True

    text = path.read_text(encoding="utf-8").rstrip()
    path.write_text(text[:-1] + "ClassAssertion(:order :i1)\n)\n", encoding="utf-8")

    assert Reasoner(str(path)).check() is False
