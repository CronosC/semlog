"""A builder produces one ontology, and says so rather than merging two."""

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.settings import DEFAULT_IRI, OntoBuildSettings, OwlProfile

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
    """The refusal is the point: without it both logs land in one ontology."""
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(_store("a"))

    with pytest.raises(ValueError):
        builder.build_ontology(_store("b"))

    names = {c.iri.get_remainder() for c in builder.onto.classes_in_signature()}
    assert "actb" not in names
    assert "acta" in names


def test_a_rejected_build_leaves_the_builder_usable():
    """Nothing was written, so the instance is not spent."""
    settings = OntoBuildSettings(profile=OwlProfile.OWL2EL)
    settings.order_representation = list(settings.order_representation)
    builder = OCEDOntoBuilder(NS, settings)

    # Force a validation failure, then repair it.
    settings.bin_labels = ["only_one"]
    with pytest.raises(ValueError, match="bin_labels"):
        builder.build_ontology(_store("a"))

    settings.bins = 1
    semlog = builder.build_ontology(_store("a"))

    assert list(semlog.onto.classes_in_signature())


def test_settings_mutated_mid_build_do_not_change_the_result():
    """Everything reads settings live, so a build has to take its own copy."""
    settings = OntoBuildSettings()
    builder = OCEDOntoBuilder(NS, settings)
    semlog = builder.build_ontology(_store("a"))

    settings.derive_annotations = True
    settings.case_concepts.append("sneaky")

    assert semlog.settings.derive_annotations is False
    assert semlog.settings.case_concepts == []


def test_snapshot_is_independent_of_the_original():
    settings = OntoBuildSettings(case_concepts=["order"])
    clone = settings.snapshot()

    settings.case_concepts.append("item")
    settings.profile = OwlProfile.OWL2EL

    assert clone.case_concepts == ["order"]
    assert clone.profile is OwlProfile.OWL2DL


def test_to_dict_does_not_read_from_disk(tmp_path):
    """Serialising must not depend on when it happens."""
    missing = tmp_path / "absent.json"
    settings = OntoBuildSettings(annotation_document=str(missing))

    # No FileNotFoundError: to_dict reports what is resolved, it does not resolve.
    assert settings.to_dict()["annotation_data"] is None
    assert settings.to_dict()["annotation_document"] == str(missing)


def test_resolve_is_what_reads_the_files(tmp_path):
    import json

    document = tmp_path / "annotations.json"
    document.write_text(json.dumps({"Event": {"label": "Occurrence"}}))
    settings = OntoBuildSettings(annotation_document=str(document))

    assert settings.to_dict()["annotation_data"] is None
    settings.resolve()
    assert settings.to_dict()["annotation_data"] == {"Event": {"label": "Occurrence"}}
