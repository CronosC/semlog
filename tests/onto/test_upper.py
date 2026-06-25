import pytest
from owlapy.iri import IRI
from owlapy.owl_ontology import OWLClass
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty

from semanticlog.onto.settings import InverseRepresentation, OntoBuildSettings
from semanticlog.onto.upper import OCEDUpperOntoBuilder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_IRI = "https://test.example.org/upper#"

TOP_LEVEL_CLASSES = ["Object", "Event", "Attribute", "Time"]
SUB_ATTRIBUTE_CLASSES = ["ObjectAttribute", "EventAttribute"]
CORE_OBJECT_PROPERTIES = [
    "observed_at",
    "event_object_relation",
    "object_object_relation",
    "has_attribute",
]
NAMED_INVERSES = ["observes", "attribute_of"]
AUTO_NAMED_INVERSES = ["inv_event_object_relation", "inv_object_object_relation"]
DATA_PROPERTIES = ["has_data_value"]


def _owl_class(name: str) -> OWLClass:
    return OWLClass(IRI(TEST_IRI, name))


def _obj_prop(name: str) -> OWLObjectProperty:
    return OWLObjectProperty(IRI(TEST_IRI, name))


def _data_prop(name: str) -> OWLDataProperty:
    return OWLDataProperty(IRI(TEST_IRI, name))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def oced_with_inverses(tmp_path):
    s = OntoBuildSettings()
    s.inverse_representation = [
        InverseRepresentation.AXIOMATIC,
        InverseRepresentation.IMPLICIT,
    ]
    b = OCEDUpperOntoBuilder(TEST_IRI, s)
    b.construct_upper_ontology(str(tmp_path / "upper.owl"))
    return b


@pytest.fixture
def oced_without_inverses(tmp_path):
    s = OntoBuildSettings()
    s.inverse_representation = []
    b = OCEDUpperOntoBuilder(TEST_IRI, s)
    b.construct_upper_ontology(str(tmp_path / "upper.owl"))
    return b


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def obj_prop_names(builder) -> set[str]:
    return {p.iri.remainder for p in builder.onto.object_properties_in_signature()}


# ---------------------------------------------------------------------------
# Tests — classes
# ---------------------------------------------------------------------------


def test_top_level_classes_all_present(oced_with_inverses):
    classes = list(oced_with_inverses.onto.classes_in_signature())
    for name in TOP_LEVEL_CLASSES:
        assert _owl_class(name) in classes, f"Expected class '{name}' in signature"


def test_sub_attribute_classes_present(oced_with_inverses):
    classes = list(oced_with_inverses.onto.classes_in_signature())
    for name in SUB_ATTRIBUTE_CLASSES:
        assert _owl_class(name) in classes, f"Expected class '{name}' in signature"


def test_class_count_is_six(oced_with_inverses):
    assert len(list(oced_with_inverses.onto.classes_in_signature())) == 6


# ---------------------------------------------------------------------------
# Tests — core properties
# ---------------------------------------------------------------------------


def test_core_object_properties_present(oced_with_inverses):
    obj_props = list(oced_with_inverses.onto.object_properties_in_signature())
    for name in CORE_OBJECT_PROPERTIES:
        assert _obj_prop(name) in obj_props, f"Expected object property '{name}'"


def test_data_property_has_data_value_present(oced_with_inverses):
    data_props = list(oced_with_inverses.onto.data_properties_in_signature())
    assert _data_prop("has_data_value") in data_props


# ---------------------------------------------------------------------------
# Tests — inverse properties (enabled)
# ---------------------------------------------------------------------------


def test_named_inverse_observes_present(oced_with_inverses):
    assert "observes" in obj_prop_names(oced_with_inverses)


def test_named_inverse_attribute_of_present(oced_with_inverses):
    assert "attribute_of" in obj_prop_names(oced_with_inverses)


def test_auto_named_inverses_present(oced_with_inverses):
    names = obj_prop_names(oced_with_inverses)
    for name in AUTO_NAMED_INVERSES:
        assert name in names, f"Expected auto-named inverse '{name}'"


def test_object_property_count_with_inverses(oced_with_inverses):
    # 4 core + 4 inverses (2 named + 2 auto-named)
    assert len(list(oced_with_inverses.onto.object_properties_in_signature())) == 8


# ---------------------------------------------------------------------------
# Tests — inverse properties (disabled)
# ---------------------------------------------------------------------------


def test_no_inverse_properties_when_disabled(oced_without_inverses):
    names = obj_prop_names(oced_without_inverses)
    for name in NAMED_INVERSES + AUTO_NAMED_INVERSES:
        assert name not in names, f"Unexpected inverse property '{name}' when disabled"


def test_object_property_count_without_inverses(oced_without_inverses):
    assert len(list(oced_without_inverses.onto.object_properties_in_signature())) == 4


def test_data_property_count_without_inverses(oced_without_inverses):
    assert len(list(oced_without_inverses.onto.data_properties_in_signature())) == 1


# ---------------------------------------------------------------------------
# Tests — file output
# ---------------------------------------------------------------------------


def test_save_creates_file(tmp_path):
    s = OntoBuildSettings()
    b = OCEDUpperOntoBuilder(TEST_IRI, s)
    out = tmp_path / "upper.owl"
    b.construct_upper_ontology(str(out))
    assert out.exists()


def test_saved_file_is_nonempty(tmp_path):
    s = OntoBuildSettings()
    b = OCEDUpperOntoBuilder(TEST_IRI, s)
    out = tmp_path / "upper.owl"
    b.construct_upper_ontology(str(out))
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Tests — builder configuration
# ---------------------------------------------------------------------------


def test_iri_ends_with_hash(oced_with_inverses):
    assert oced_with_inverses.iri.endswith("#")


# ---------------------------------------------------------------------------
# Tests — OcelExtension: ATTRIBUTE_HISTORY
# ---------------------------------------------------------------------------


@pytest.fixture
def oced_with_attribute_history(tmp_path):
    from semanticlog.onto.settings import OcelExtension

    s = OntoBuildSettings(ocel_extensions=[OcelExtension.ATTRIBUTE_HISTORY])
    s.inverse_representation = []
    b = OCEDUpperOntoBuilder(TEST_IRI, s)
    b.construct_upper_ontology(str(tmp_path / "upper_ah.owl"))
    return b


def test_attribute_history_adds_attribute_change_class(oced_with_attribute_history):
    classes = list(oced_with_attribute_history.onto.classes_in_signature())
    assert _owl_class("AttributeChange") in classes


def test_attribute_history_adds_has_attribute_change_property(
    oced_with_attribute_history,
):
    assert "has_attribute_change" in obj_prop_names(oced_with_attribute_history)


def test_attribute_history_adds_attribute_change_value_property(
    oced_with_attribute_history,
):
    assert "attribute_change_value" in obj_prop_names(oced_with_attribute_history)


def test_attribute_history_adds_changed_at_property(oced_with_attribute_history):
    assert "changed_at" in obj_prop_names(oced_with_attribute_history)


def test_no_extension_does_not_add_attribute_change_class(oced_with_inverses):
    classes = list(oced_with_inverses.onto.classes_in_signature())
    assert _owl_class("AttributeChange") not in classes
