"""Round-trip tests for SemanticLog serialisation."""

import pandas as pd
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual

from semanticlog.data.oced import OCEDStore
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.semlog import SemanticLog
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    InverseRepresentation,
    OntoBuildSettings,
    OwlProfile,
)


def _minimal_oced() -> OCEDStore:
    oced = OCEDStore()
    oced.add_event(
        "act",
        "eid",
        "ts",
        pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2022-01-01"], utc=True)}),
    )
    oced.add_object("item", "oid", pd.DataFrame({"oid": ["o1"]}))
    return oced


def _minimal_settings() -> OntoBuildSettings:
    s = OntoBuildSettings()
    s.profile = OwlProfile.OWL2EL
    s.order_representation = []
    s.case_concepts = []
    s.inverse_representation = [InverseRepresentation.IMPLICIT]
    return s


def test_semlog_write_ontology(tmp_path):
    """write_ontology writes a loadable OWL file."""
    oced = _minimal_oced()
    semlog = OCEDOntoBuilder(DEFAULT_IRI, _minimal_settings()).build_ontology(oced)

    path = str(tmp_path / "out.owl")
    semlog.write_ontology(path)

    from owlapy.owl_ontology import Ontology

    onto = Ontology(IRI.create(f"file://{path}"), load=True)
    assert len(list(onto.classes_in_signature())) > 0


def test_semlog_pickle_round_trip(tmp_path):
    """write + read preserves classes and individuals."""
    oced = _minimal_oced()
    semlog = OCEDOntoBuilder(DEFAULT_IRI, _minimal_settings()).build_ontology(oced)

    path = str(tmp_path / "out.semlog")
    semlog.write(path)

    loaded = SemanticLog.read(path)

    original_classes = {c.iri.remainder for c in semlog.onto.classes_in_signature()}
    loaded_classes = {c.iri.remainder for c in loaded.onto.classes_in_signature()}
    assert original_classes == loaded_classes

    original_inds = {i.iri.remainder for i in semlog.onto.individuals_in_signature()}
    loaded_inds = {i.iri.remainder for i in loaded.onto.individuals_in_signature()}
    assert original_inds == loaded_inds


def test_semlog_pickle_preserves_settings(tmp_path):
    """Settings are preserved through the pickle round-trip."""
    oced = _minimal_oced()
    settings = _minimal_settings()
    semlog = OCEDOntoBuilder(DEFAULT_IRI, settings).build_ontology(oced)

    path = str(tmp_path / "out.semlog")
    semlog.write(path)
    loaded = SemanticLog.read(path)

    assert loaded.settings.profile == settings.profile
    assert loaded.settings.case_concepts == settings.case_concepts
    assert loaded.iri == DEFAULT_IRI


def test_semlog_pickle_individuals_accessible(tmp_path):
    """Individuals from the original ontology are accessible after round-trip."""
    oced = _minimal_oced()
    semlog = OCEDOntoBuilder(DEFAULT_IRI, _minimal_settings()).build_ontology(oced)

    path = str(tmp_path / "out.semlog")
    semlog.write(path)
    loaded = SemanticLog.read(path)

    ind = OWLNamedIndividual(IRI(DEFAULT_IRI, "e1"))
    assert ind in list(loaded.onto.individuals_in_signature())
