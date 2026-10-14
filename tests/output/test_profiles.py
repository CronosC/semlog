"""Whether a build stays in the OWL 2 profile its settings name, as the OWL API checks it."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.core.enums import OWLProfile
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind
from semanticlog.onto.settings import OntoBuildSettings
from tests.support.build import BACKENDS, STREAMING, emit
from tests.support.owlapi import profile_violations
from tests.support.settings import MATRIX, NAMES

_DAY = datetime(2024, 1, 1, 9, 0)


def _report(name: str, backend: str, profile: str, violations: dict) -> str:
    """Name the sorts of violation, with the start of one example of each."""
    lines = [
        f"  {kind}: {len(found)}\n      {found[0][:160]}"
        for kind, found in sorted(violations.items())
    ]
    return (
        f"{name} through the {backend} sink is built for {profile} "
        f"and is not in it:\n" + "\n".join(lines)
    )


@pytest.mark.reasoner
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", NAMES)
def test_every_build_is_in_the_profile_its_settings_name(name, backend, built_path):
    profile = MATRIX[name]().settings.profile
    violations = profile_violations(built_path(name, backend), profile)
    assert not violations, _report(name, backend, profile.value, violations)


def _boolean_store() -> OCEDStore:
    """One object type carrying a boolean attribute, which the matrix store lacks."""
    data = OCEDStore()
    data.add(
        EntityKind.OBJECT,
        "order",
        pd.DataFrame({"oid": ["o1", "o2"], "express": [True, False]}),
        id_col="oid",
    )
    data.add(
        EntityKind.EVENT,
        "place_order",
        pd.DataFrame({"eid": ["e1", "e2"], "ts": [_DAY, _DAY]}),
        id_col="eid",
        time_col="ts",
    )
    data.add_relation(
        RelationKind.E2O,
        "concerns",
        pd.DataFrame({"eid": ["e1", "e2"], "oid": ["o1", "o2"]}),
        from_id="eid",
        to_id="oid",
        to_type=["order"],
    )
    return clean_oced_data(data)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_boolean_stays_inside_the_el_datatype_map(backend, tmp_path):
    """`xsd:boolean` is in the OWL 2 DL datatype map and not in the EL one."""
    suffix = "ofn" if backend == STREAMING else "owl"
    written = {}
    for profile in (OWLProfile.OWL2DL, OWLProfile.OWL2EL):
        path = tmp_path / f"{profile.value}.{backend}.{suffix}"
        emit(
            OntoBuildSettings(profile=profile),
            path,
            backend,
            data=_boolean_store(),
        )
        written[profile] = path.read_text(encoding="utf-8")

    assert "boolean" in written[OWLProfile.OWL2DL]
    assert "boolean" not in written[OWLProfile.OWL2EL]
