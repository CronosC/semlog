"""A whole build, checked against the ontology it produces."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from owlapy.class_expression import OWLClass
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_ontology import Ontology
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty
from owlapy.owl_reasoner import StructuralReasoner

from semanticlog.core import vocabulary as vocab
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import (
    OCEDStore,
)
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OntoBuildSettings,
    OWLProfile,
)

NS = DEFAULT_IRI


@pytest.fixture
def settings() -> OntoBuildSettings:
    return OntoBuildSettings(
        profile=OWLProfile.OWL2EL,
        inverse_representations=[InverseRepresentation.IMPLICIT],
        order_representations=[],
        case_concepts=[],
    )


@pytest.fixture
def store() -> OCEDStore:
    data = OCEDStore()
    data.add_event(
        "activity1",
        id_col="id",
        time_col="time",
        infer=False,
        data=pd.DataFrame(
            {
                "time": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)],
                "id": ["e" + str(i) for i in range(10)],
                "resource": [
                    "Bob",
                    "Bob",
                    "Alice",
                    "Bob",
                    "Alice",
                    "Alice",
                    "Alice",
                    "Bob",
                    "Alice",
                    "Alice",
                ],
                "number": [
                    10.0,
                    26.5,
                    50.1,
                    23.2,
                    50.3,
                    65.4,
                    72.5,
                    46.6,
                    96.7,
                    16.8,
                ],
            }
        ),
    )
    data.add_object(
        "thing1",
        id_col="id",
        infer=False,
        data=pd.DataFrame(
            {
                "id": ["o" + str(i) for i in range(10)],
                "type": [
                    "Small",
                    "Small",
                    "Medium",
                    "Large",
                    "Large",
                    "Large",
                    "Small",
                    "Medium",
                    "Small",
                    "Medium",
                ],
                "value": [
                    120.0,
                    None,
                    605.4,
                    243.2,
                    530.3,
                    605.4,
                    732.5,
                    416.6,
                    196.7,
                    136.8,
                ],
            }
        ),
    )
    data.add_event_to_object_relation(
        "does",
        from_id_col="event_id",
        to_id_col="object_id",
        to_type=["thing1"],
        data=pd.DataFrame(
            {
                "event_id": [f"e{i}" for i in range(10)],
                "object_id": [f"o{i}" for i in range(10)],
            }
        ),
    )
    data.add_object_to_object_relation(
        "connects",
        from_id_col="object_id1",
        from_type=["thing1"],
        to_id_col="object_id2",
        to_type=["thing1"],
        data=pd.DataFrame(
            {
                "object_id1": [f"o{i}" for i in range(10)],
                "object_id2": [f"o{(i + 1) % 10}" for i in range(10)],
            }
        ),
    )
    return clean_oced_data(data)


@pytest.fixture
def owl_path(tmp_path) -> str:
    return str(tmp_path / "tmp.owl")


def load_reasoner(owl_path: str) -> tuple[Ontology, StructuralReasoner]:
    onto = Ontology(IRI.create(f"file://{owl_path}"))
    return onto, StructuralReasoner(onto)


def test_events_basic(settings, store, owl_path):
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.events.emit(store)
    builder.sink.save(owl_path)

    onto, reasoner = load_reasoner(owl_path)

    activity1 = OWLClass(IRI(NS, "activity1"))
    assert activity1 in list(onto.classes_in_signature())

    expected_inds = {OWLNamedIndividual(IRI(NS, f"e{i}")) for i in range(10)}
    assert set(reasoner.instances(activity1)) == expected_inds, (
        "activity1 instances do not match the 10 event individuals e0..e9"
    )


def test_objects_basic(settings, store, owl_path):
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.objects.emit(store)
    builder.sink.save(owl_path)

    onto, reasoner = load_reasoner(owl_path)

    thing1 = OWLClass(IRI(NS, "thing1"))
    assert thing1 in list(onto.classes_in_signature())

    expected_inds = {OWLNamedIndividual(IRI(NS, f"o{i}")) for i in range(10)}
    assert set(reasoner.instances(thing1)) == expected_inds, (
        "thing1 instances do not match the 10 object individuals o0..o9"
    )


def test_e2o_relations(settings, store, owl_path):
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.events.emit(store)
    builder.objects.emit(store)
    builder.relations.emit_event_to_object(store)
    builder.sink.save(owl_path)

    onto, reasoner = load_reasoner(owl_path)

    does = OWLObjectProperty(IRI(NS, "does"))
    assert does in list(onto.object_properties_in_signature())

    rel = store.get_relation("does")
    from_key, to_key = rel.from_id, rel.to_id
    expected_pairs = rel.data[[from_key, to_key]].dropna().drop_duplicates()
    for _, row in expected_pairs.iterrows():
        from_ind = OWLNamedIndividual(IRI(NS, str(row[from_key])))
        to_ind = OWLNamedIndividual(IRI(NS, str(row[to_key])))
        targets = set(reasoner.object_property_values(from_ind, does, direct=False))
        assert to_ind in targets, (
            f"Missing 'does' assertion: {row[from_key]} → {row[to_key]}"
        )


def test_o2o_relations(settings, store, owl_path):
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.events.emit(store)
    builder.objects.emit(store)
    builder.relations.emit_object_to_object(store)
    builder.sink.save(owl_path)

    onto, reasoner = load_reasoner(owl_path)

    connects = OWLObjectProperty(IRI(NS, "connects"))
    assert connects in list(onto.object_properties_in_signature())

    rel = store.get_relation("connects")
    from_key, to_key = rel.from_id, rel.to_id
    expected_pairs = rel.data[[from_key, to_key]].dropna().drop_duplicates()
    for _, row in expected_pairs.iterrows():
        from_ind = OWLNamedIndividual(IRI(NS, str(row[from_key])))
        to_ind = OWLNamedIndividual(IRI(NS, str(row[to_key])))
        targets = set(reasoner.object_property_values(from_ind, connects, direct=False))
        assert to_ind in targets, (
            f"Missing 'connects' assertion: {row[from_key]} → {row[to_key]}"
        )


def test_time_basic(settings, store, owl_path):
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.times.emit(store)
    builder.sink.save(owl_path)

    onto, reasoner = load_reasoner(owl_path)

    time_cls = OWLClass(IRI(NS, "Time"))
    assert time_cls in list(onto.classes_in_signature())

    assert len(list(reasoner.instances(time_cls))) == 10, (
        "Expected exactly 10 Time individuals for 10 unique timestamps"
    )

    observed_at = OWLObjectProperty(IRI(NS, "observed_at"))
    assert observed_at in list(onto.object_properties_in_signature())


def test_build_all(settings, store, owl_path):
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.build_ontology(store)
    builder.sink.save(owl_path)

    onto, reasoner = load_reasoner(owl_path)

    activity1 = OWLClass(IRI(NS, "activity1"))
    thing1 = OWLClass(IRI(NS, "thing1"))
    time_cls = OWLClass(IRI(NS, "Time"))

    declared = list(onto.classes_in_signature())
    for cls in [activity1, thing1, time_cls]:
        assert cls in declared, f"class missing: {cls}"

    assert list(reasoner.instances(activity1)), "No instances for activity1"
    assert list(reasoner.instances(thing1)), "No instances for thing1"
    assert list(reasoner.instances(time_cls)), "No instances for Time"


def test_declare_requires_file(settings, store):
    settings = settings.replace(
        order_representations=[EventOrderRepresentation.DECLARE],
        declare_file=None,
    )
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)

    with pytest.raises(ValueError):
        builder.order.emit(store)


def test_declare_single_case(settings, store, owl_path, tmp_path):
    df = pd.DataFrame(
        {
            "case_concept": ["thing1", "thing1", "thing1"],
            "trace_id": ["o1", "o3", "o7"],
            "constraintA": [True, False, True],
            "constraintB": [False, True, False],
        }
    )
    declare_path = tmp_path / "df_declare_testing.csv"
    df.to_csv(declare_path, index=False)

    settings = settings.replace(
        order_representations=[EventOrderRepresentation.DECLARE],
        declare_file=str(declare_path),
    )

    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.order.emit(store)
    builder.sink.save(owl_path)

    onto, reasoner = load_reasoner(owl_path)
    ns = DEFAULT_IRI

    declared = list(onto.classes_in_signature())
    assert OWLClass(IRI(ns, "DeclareConstraint")) in declared
    constraintA = OWLClass(IRI(ns, "constraintA"))
    constraintB = OWLClass(IRI(ns, "constraintB"))
    assert constraintA in declared
    assert constraintB in declared

    ind1 = OWLNamedIndividual(IRI(ns, "o1"))
    ind2 = OWLNamedIndividual(IRI(ns, "o3"))
    ind3 = OWLNamedIndividual(IRI(ns, "o7"))

    in_a = list(reasoner.instances(constraintA))
    assert ind1 in in_a
    assert ind3 in in_a
    assert ind2 not in in_a

    in_b = list(reasoner.instances(constraintB))
    assert ind2 in in_b
    assert ind1 not in in_b
    assert ind3 not in in_b


def test_declare_multi_case(settings, store, owl_path, tmp_path):
    df = pd.DataFrame(
        {
            "case_concept": ["thing1", "thing2", "thing1", "thing2"],
            "trace_id": ["o1", "o2", "o3", "o4"],
            "constraintA": [True, np.nan, True, np.nan],
            "constraintB": [np.nan, True, np.nan, True],
        }
    )
    declare_path = tmp_path / "df_declare_multi_case.csv"
    df.to_csv(declare_path, index=False)

    settings = settings.replace(
        order_representations=[EventOrderRepresentation.DECLARE],
        declare_file=str(declare_path),
    )

    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.order.emit(store)
    builder.sink.save(owl_path)

    onto, reasoner = load_reasoner(owl_path)
    ns = DEFAULT_IRI

    declared = list(onto.classes_in_signature())
    assert OWLClass(IRI(ns, "DeclareConstraint")) in declared
    constraintA = OWLClass(IRI(ns, "constraintA"))
    constraintB = OWLClass(IRI(ns, "constraintB"))
    assert constraintA in declared
    assert constraintB in declared

    ind_o1 = OWLNamedIndividual(IRI(ns, "o1"))
    ind_o2 = OWLNamedIndividual(IRI(ns, "o2"))
    ind_o3 = OWLNamedIndividual(IRI(ns, "o3"))
    ind_o4 = OWLNamedIndividual(IRI(ns, "o4"))

    in_a = list(reasoner.instances(constraintA))
    assert ind_o1 in in_a
    assert ind_o3 in in_a
    assert ind_o2 not in in_a
    assert ind_o4 not in in_a

    in_b = list(reasoner.instances(constraintB))
    assert ind_o2 in in_b
    assert ind_o4 in in_b
    assert ind_o1 not in in_b
    assert ind_o3 not in in_b


def test_class_hierarchy(settings, store, owl_path):
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.events.emit(store)
    builder.objects.emit(store)
    builder.sink.save(owl_path)

    onto, reasoner = load_reasoner(owl_path)

    event_cls = OWLClass(IRI(NS, "Event"))
    object_cls = OWLClass(IRI(NS, "Object"))
    activity1 = OWLClass(IRI(NS, "activity1"))
    thing1 = OWLClass(IRI(NS, "thing1"))

    declared = list(onto.classes_in_signature())
    for cls in [event_cls, object_cls, activity1, thing1]:
        assert cls in declared, f"class missing: {cls}"

    assert event_cls in set(reasoner.super_classes(activity1, direct=False)), (
        "activity1 must be a subclass of Event"
    )
    assert object_cls in set(reasoner.super_classes(thing1, direct=False)), (
        "thing1 must be a subclass of Object"
    )


def test_numeric_attrs_value_nodes(settings, store, owl_path):
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.objects.emit(store)
    builder.sink.save(owl_path)

    onto, reasoner = load_reasoner(owl_path)

    value_cls = OWLClass(IRI(NS, "value"))
    assert value_cls in list(onto.classes_in_signature())

    inds = set(onto.individuals_in_signature())
    expected_inds = [
        "value_120.0",
        "value_605.4",
        "value_243.2",
        "value_530.3",
        "value_732.5",
        "value_416.6",
        "value_196.7",
        "value_136.8",
    ]
    for name in expected_inds:
        assert OWLNamedIndividual(IRI(NS, name)) in inds, f"Missing {name}"

    has_value_prop = OWLObjectProperty(IRI(NS, vocab.has_attribute_property("value")))
    assert has_value_prop in list(onto.object_properties_in_signature())

    has_data_value = OWLDataProperty(IRI(NS, "has_data_value"))
    assert has_data_value in list(onto.data_properties_in_signature())

    o2_ind = OWLNamedIndividual(IRI(NS, "o2"))
    o5_ind = OWLNamedIndividual(IRI(NS, "o5"))
    v605 = OWLNamedIndividual(IRI(NS, "value_605.4"))

    assert v605 in set(
        reasoner.object_property_values(o2_ind, has_value_prop, direct=False)
    )
    assert v605 in set(
        reasoner.object_property_values(o5_ind, has_value_prop, direct=False)
    )

    literals = list(reasoner.data_property_values(v605, has_data_value, direct=False))
    assert len(literals) == 1, (
        "Expected exactly one has_data_value literal on value_605.4"
    )
    assert "605.4" in str(literals[0]), "Literal value does not contain 605.4"


def test_string_attrs_value_nodes(settings, store, owl_path):
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.events.emit(store)
    builder.sink.save(owl_path)

    onto, reasoner = load_reasoner(owl_path)

    resource_cls = OWLClass(IRI(NS, "resource"))
    assert resource_cls in list(onto.classes_in_signature())

    alice_ind = OWLNamedIndividual(IRI(NS, "resource_Alice"))
    bob_ind = OWLNamedIndividual(IRI(NS, "resource_Bob"))
    inds = set(onto.individuals_in_signature())
    assert alice_ind in inds, "Expected individual 'resource_Alice' not found"
    assert bob_ind in inds, "Expected individual 'resource_Bob' not found"

    typed_as_resource = set(reasoner.instances(resource_cls))
    assert alice_ind in typed_as_resource
    assert bob_ind in typed_as_resource

    has_resource_prop = OWLObjectProperty(IRI(NS, "has_resource"))
    assert has_resource_prop in list(onto.object_properties_in_signature())

    expected_resource = {
        "e0": bob_ind,
        "e1": bob_ind,
        "e2": alice_ind,
        "e3": bob_ind,
        "e4": alice_ind,
        "e5": alice_ind,
        "e6": alice_ind,
        "e7": bob_ind,
        "e8": alice_ind,
        "e9": alice_ind,
    }
    for eid, expected_ind in expected_resource.items():
        ev_ind = OWLNamedIndividual(IRI(NS, eid))
        targets = set(
            reasoner.object_property_values(ev_ind, has_resource_prop, direct=False)
        )
        assert targets == {expected_ind}, (
            f"Event {eid}: expected resource {expected_ind}, got {targets}"
        )


def test_bin_labels_mismatch_raises():
    with pytest.raises(ValueError):
        OntoBuildSettings(bins=3, bin_labels=["low", "high"])


def test_data_representation_none_skips_numeric(settings, store, owl_path):
    settings = settings.replace(data_representations=[])
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.objects.emit(store)
    builder.sink.save(owl_path)

    onto, _ = load_reasoner(owl_path)

    classes = set(onto.classes_in_signature())
    assert OWLClass(IRI(NS, "value")) not in classes

    inds = set(onto.individuals_in_signature())
    labels = ["very_low", "low", "average", "high", "very_high"]
    for label in labels:
        assert OWLNamedIndividual(IRI(NS, f"value_{label}")) not in inds
    for name in ["value_120.0", "value_605.4", "value_243.2"]:
        assert OWLNamedIndividual(IRI(NS, name)) not in inds

    obj_props = set(onto.object_properties_in_signature())
    assert OWLObjectProperty(IRI(NS, "has_value")) not in obj_props

    assert OWLClass(IRI(NS, "type")) in classes, "string attributes unaffected"


def test_data_representation_values_has_data_value(settings, store, owl_path):
    settings = settings.replace(data_representations=[DataRepresentation.VALUES])
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.objects.emit(store)
    builder.sink.save(owl_path)

    onto, reasoner = load_reasoner(owl_path)

    has_data_value = OWLDataProperty(IRI(NS, "has_data_value"))
    assert has_data_value in list(onto.data_properties_in_signature())

    v605 = OWLNamedIndividual(IRI(NS, "value_605.4"))
    literals = list(reasoner.data_property_values(v605, has_data_value, direct=False))
    assert len(literals) == 1, (
        "Expected exactly one has_data_value literal on value_605.4"
    )
    assert "605.4" in str(literals[0]), "Literal does not contain 605.4"


def test_data_representation_equal_bins_produces_labels(settings, store, owl_path):
    settings = settings.replace(data_representations=[DataRepresentation.EQUAL_BINS])
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.objects.emit(store)
    builder.sink.save(owl_path)

    onto, reasoner = load_reasoner(owl_path)
    inds = set(onto.individuals_in_signature())
    labels = settings.bin_labels
    value_cls = OWLClass(IRI(NS, "value"))

    class_names = {c.iri.remainder for c in onto.classes_in_signature()}
    found = [lbl for lbl in labels if f"value_bin_{lbl}" in class_names]
    assert set(found) == set(labels), f"Missing bin classes: {set(labels) - set(found)}"
    for lbl in found:
        supers = set(
            reasoner.super_classes(OWLClass(IRI(NS, f"value_bin_{lbl}")), direct=True)
        )
        assert value_cls in supers, f"value_bin_{lbl} not a subclass of 'value'"

    typed = set()
    for lbl in found:
        for inst in reasoner.instances(OWLClass(IRI(NS, f"value_bin_{lbl}"))):
            typed.add(inst.iri.remainder)
    assert len(typed) > 0, "No object individuals typed as any bin class"

    for lbl in labels:
        assert OWLNamedIndividual(IRI(NS, f"value_{lbl}")) not in inds, (
            "bin labels must not become individuals"
        )

    for name in ["value_120.0", "value_605.4", "value_243.2"]:
        assert OWLNamedIndividual(IRI(NS, name)) in inds

    for lbl in labels:
        for inst in reasoner.instances(OWLClass(IRI(NS, f"value_bin_{lbl}"))):
            assert inst.iri.remainder.startswith("value_"), (
                f"{inst.iri.remainder} is an owner typed into a bin class"
            )


def test_data_representation_dist_bins_no_literals(settings, owl_path):
    data = OCEDStore()
    rng = np.random.default_rng(42)
    n = 20
    values = rng.uniform(1, 1000, n).tolist()
    data.add_object(
        "thing1",
        id_col="id",
        infer=False,
        data=pd.DataFrame({"id": [f"o{i}" for i in range(n)], "value": values}),
    )

    settings = settings.replace(data_representations=[DataRepresentation.DIST_BINS])
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.objects.emit(data)
    builder.sink.save(owl_path)

    onto, reasoner = load_reasoner(owl_path)
    inds = set(onto.individuals_in_signature())
    labels = settings.bin_labels
    value_cls = OWLClass(IRI(NS, "value"))

    class_names = {c.iri.remainder for c in onto.classes_in_signature()}
    found = [lbl for lbl in labels if f"value_bin_{lbl}" in class_names]
    assert len(found) > 0, "No bin classes found as subclasses of 'value'"
    assert set(found).issubset(set(labels)), (
        f"Unexpected bin classes outside configured labels: {set(found) - set(labels)}"
    )
    for lbl in found:
        supers = set(
            reasoner.super_classes(OWLClass(IRI(NS, f"value_bin_{lbl}")), direct=True)
        )
        assert value_cls in supers, f"value_bin_{lbl} not a subclass of 'value'"

    typed = set()
    for lbl in found:
        for inst in reasoner.instances(OWLClass(IRI(NS, f"value_bin_{lbl}"))):
            typed.add(inst.iri.remainder)
    assert len(typed) > 0, "No object individuals typed as any bin class"

    for lbl in labels:
        assert OWLNamedIndividual(IRI(NS, f"value_{lbl}")) not in inds, (
            "bin labels must not become individuals"
        )


def test_data_representation_values_equal_bins_types_value_ind_as_bin_class(
    settings, store, owl_path
):
    settings = settings.replace(
        data_representations=[
            DataRepresentation.VALUES,
            DataRepresentation.EQUAL_BINS,
        ]
    )
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.objects.emit(store)
    builder.sink.save(owl_path)

    onto, reasoner = load_reasoner(owl_path)
    inds = set(onto.individuals_in_signature())
    labels = settings.bin_labels

    assert OWLNamedIndividual(IRI(NS, "value_120.0")) in inds
    assert OWLNamedIndividual(IRI(NS, "value_605.4")) in inds

    ind = OWLNamedIndividual(IRI(NS, "value_120.0"))
    types = set(reasoner.types(ind, direct=False))
    bin_classes = {OWLClass(IRI(NS, f"value_bin_{lbl}")) for lbl in labels}
    assert types & bin_classes, "value_120.0 is not typed as any bin class"


def test_make_inverses_explicit_emits_observes_assertions(tmp_path):
    s = OntoBuildSettings(
        profile=OWLProfile.OWL2DL,
        inverse_representations=[
            InverseRepresentation.AXIOMATIC,
            InverseRepresentation.IMPLICIT,
        ],
        order_representations=[],
        case_concepts=[],
    )

    data = OCEDStore()
    data.add_event(
        "act1",
        id_col="id",
        time_col="time",
        infer=False,
        data=pd.DataFrame(
            {
                "id": ["e0", "e1"],
                "time": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            }
        ),
    )

    abox_path = str(tmp_path / "abox.owl")
    builder = OCEDOntoBuilder(DEFAULT_IRI, s)
    builder.times.emit(data)
    builder.sink.save(abox_path)

    onto = Ontology(IRI.create(f"file://{abox_path}"))
    reasoner = StructuralReasoner(onto)

    time_ind = OWLNamedIndividual(IRI(NS, "20230101T000000_000000"))
    event_ind = OWLNamedIndividual(IRI(NS, "e0"))
    observes_prop = OWLObjectProperty(IRI(NS, "observes"))

    assert observes_prop in list(onto.object_properties_in_signature()), (
        "observes property missing from ABox signature, inverse not declared"
    )
    values = set(reasoner.object_property_values(time_ind, observes_prop, direct=False))
    assert event_ind in values, (
        f"Expected observes({time_ind.str}, {event_ind.str}) in ABox. "
        "make_inverses_explicit did not emit inverse assertion"
    )


def test_attr_prop_inverse_declared(tmp_path):
    s = OntoBuildSettings()
    s = s.replace(profile=OWLProfile.OWL2DL)
    s = s.replace(
        inverse_representations=[
            InverseRepresentation.AXIOMATIC,
            InverseRepresentation.IMPLICIT,
        ]
    )
    s = s.replace(order_representations=[])
    s = s.replace(case_concepts=[])

    data = OCEDStore()
    data.add_event(
        "act1",
        id_col="id",
        time_col="time",
        infer=False,
        data=pd.DataFrame(
            {
                "id": ["e0", "e1"],
                "time": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
                "resource": ["Alice", "Bob"],
            }
        ),
    )

    builder = OCEDOntoBuilder(DEFAULT_IRI, s)
    builder.events.emit(data)

    props = {p.iri.remainder for p in builder.onto.object_properties_in_signature()}
    assert "has_resource" in props, "has_resource must be declared"
    assert "inv_has_resource" in props, (
        "inv_has_resource must be declared when inverse_properties=True"
    )


@pytest.mark.parametrize("profile", [OWLProfile.OWL2DL, OWLProfile.OWL2EL])
def test_inv_df_subproperty_of_inv_df(tmp_path, profile):
    s = OntoBuildSettings()
    s = s.replace(profile=profile)
    s = s.replace(inverse_representations=[InverseRepresentation.IMPLICIT])
    s = s.replace(order_representations=[EventOrderRepresentation.DF])
    s = s.replace(case_concepts=["order"])

    data = OCEDStore()
    data.add_event(
        "act1",
        id_col="id",
        time_col="time",
        infer=False,
        data=pd.DataFrame(
            {
                "id": ["e0", "e1", "e2"],
                "time": [
                    datetime(2023, 1, 1),
                    datetime(2023, 1, 2),
                    datetime(2023, 1, 3),
                ],
            }
        ),
    )
    data.add_object("order", id_col="id", data=pd.DataFrame({"id": ["o1"]}))
    data.add_event_to_object_relation(
        "modifies",
        from_id_col="from_id",
        to_id_col="to_id",
        to_type=["order"],
        data=pd.DataFrame({"from_id": ["e0", "e1", "e2"], "to_id": ["o1", "o1", "o1"]}),
    )

    builder = OCEDOntoBuilder(DEFAULT_IRI, s)
    builder.build_ontology(data)

    props = {p.iri.remainder for p in builder.onto.object_properties_in_signature()}
    assert "inv_df" in props, "inv_df must be declared"
    assert "inv_df_order" in props, "inv_df_order must be declared"

    path = str(tmp_path / "inverses.owl")
    builder.sink.save(path)
    onto, reasoner = load_reasoner(path)
    parents = {
        p.iri.remainder
        for p in reasoner.super_object_properties(
            OWLObjectProperty(IRI(NS, "inv_df_order")), direct=True
        )
    }
    assert "inv_df" in parents, f"inv_df_order's parents are {parents}"
