"""The build result, and reasoning over it."""

import pandas as pd
import pytest

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    InverseRepresentation,
    OntoBuildSettings,
    OWLProfile,
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
    return clean_oced_data(oced)


def _minimal_settings() -> OntoBuildSettings:
    s = OntoBuildSettings()
    s = s.replace(profile=OWLProfile.OWL2EL)
    s = s.replace(order_representations=[])
    s = s.replace(case_concepts=[])
    s = s.replace(inverse_representations=[InverseRepresentation.IMPLICIT])
    return s


@pytest.mark.reasoner
def test_check_reports_consistency_without_a_destination(tmp_path):
    semlog = OCEDOntoBuilder(DEFAULT_IRI, _minimal_settings()).build_ontology(
        _minimal_oced()
    )

    assert semlog.check() is True


@pytest.mark.reasoner
def test_reason_writes_the_ontology_it_reasons_over(tmp_path):
    path = tmp_path / "reasoned.owl"
    semlog = OCEDOntoBuilder(DEFAULT_IRI, _minimal_settings()).build_ontology(
        _minimal_oced()
    )

    reasoner = semlog.reason(path=str(path))

    assert path.exists()
    assert reasoner.check() is True


def test_reason_without_a_path_explains_itself():
    semlog = OCEDOntoBuilder(DEFAULT_IRI, _minimal_settings()).build_ontology(
        _minimal_oced()
    )

    with pytest.raises(ValueError, match="only in memory"):
        semlog.reason()
