"""A builder produces one ontology and refuses a second."""

from dataclasses import FrozenInstanceError
from datetime import datetime

import pandas as pd
import pytest

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    EventOrderRepresentation,
    OntoBuildSettings,
    OWLProfile,
)

NS = DEFAULT_IRI


def _store(tag: str) -> OCEDStore:
    data = OCEDStore()
    data.add_event(
        f"act{tag}",
        id_col="eid",
        time_col="t",
        data=pd.DataFrame({"eid": [f"e{tag}"], "t": [datetime(2024, 1, 1)]}),
    )
    data.add_object(
        f"order{tag}", id_col="oid", data=pd.DataFrame({"oid": [f"o{tag}"]})
    )
    data.add_event_to_object_relation(
        f"rel{tag}",
        from_id_col="eid",
        to_id_col="oid",
        to_type=[f"order{tag}"],
        data=pd.DataFrame({"eid": [f"e{tag}"], "oid": [f"o{tag}"]}),
    )
    return clean_oced_data(data)


def test_second_build_is_refused():
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(_store("a"))

    with pytest.raises(ValueError, match="already produced an ontology"):
        builder.build_ontology(_store("b"))


def test_second_build_does_not_merge_the_two_logs():
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(_store("a"))

    with pytest.raises(ValueError):
        builder.build_ontology(_store("b"))

    names = {c.iri.remainder for c in builder.onto.classes_in_signature()}
    assert "actb" not in names
    assert "acta" in names


def test_a_build_rejected_on_its_input_leaves_the_builder_usable():
    settings = OntoBuildSettings(
        case_concepts=["order"],
        order_representations=[EventOrderRepresentation.DECLARE],
    )
    builder = OCEDOntoBuilder(NS, settings)

    with pytest.raises(ValueError, match="no DECLARE constraints are available"):
        builder.build_ontology(_store("a"))

    usable = OCEDOntoBuilder(NS, OntoBuildSettings())
    assert list(usable.build_ontology(_store("a")).onto.classes_in_signature())


def test_a_build_rejected_on_a_name_spends_the_builder():
    builder = OCEDOntoBuilder(NS, OntoBuildSettings(profile=OWLProfile.OWL2EL))

    dirty = OCEDStore()
    dirty.add_event(
        "bad name",
        id_col="eid",
        time_col="t",
        data=pd.DataFrame({"eid": ["e1"], "t": [datetime(2024, 1, 1)]}),
    )
    with pytest.raises(ValueError, match="cannot be an IRI local name"):
        builder.build_ontology(dirty)

    with pytest.raises(ValueError, match="already produced an ontology"):
        builder.build_ontology(_store("a"))


def test_settings_cannot_be_changed_after_a_build():
    settings = OntoBuildSettings(case_concepts=["order"])
    semlog = OCEDOntoBuilder(NS, settings).build_ontology(_store("a"))

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


def test_to_dict_does_not_read_from_disk(tmp_path):
    import json

    document = tmp_path / "annotations.json"
    document.write_text(json.dumps({"Event": {"label": "Occurrence"}}))
    settings = OntoBuildSettings(annotation_document=str(document))
    document.unlink()

    assert settings.to_dict()["annotation_data"] == {"Event": {"label": "Occurrence"}}


def test_a_missing_annotation_document_fails_at_construction(tmp_path):
    with pytest.raises(FileNotFoundError):
        OntoBuildSettings(annotation_document=str(tmp_path / "absent.json"))
