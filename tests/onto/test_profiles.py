"""Whether a build stays in the OWL 2 profile its settings name.

`profile` is a promise about the emitted file, and the OWL API is what can say
whether it was kept. The constructor check in `test_build_invariants` reads the
axiom lines, so it sees a forbidden class expression but not a datatype outside
the profile or an entity used without a declaration.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.core.enums import OWLProfile
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.settings import OntoBuildSettings
from tests.conftest import BACKENDS, STREAMING, emit
from tests.support.owlapi import profile_violations
from tests.support.settings import CONFIGURATIONS, NAMES

_DAY = datetime(2024, 1, 1, 9, 0)


def _report(name: str, backend: str, profile: str, violations: dict) -> str:
    """Name the sorts of violation, with one example of each.

    A violation quotes the whole axiom it came from, and a build configuration
    reaches four figures, so the example is cut.
    """
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
def test_every_build_is_in_the_profile_it_claims(name, backend, built_path):
    profile = CONFIGURATIONS[name]().settings.profile
    violations = profile_violations(built_path(name, backend), profile)
    assert not violations, _report(name, backend, profile.value, violations)


def _boolean_store() -> OCEDStore:
    """One object type carrying a boolean attribute.

    The matrix store has no boolean column, and a boolean is the value type
    the two datatype maps disagree about that a build can actually reach: it
    is numeric to pandas, so it keeps its literal, where a bare `date` column
    is object dtype and never carries one.
    """
    data = OCEDStore()
    data.add_object(
        "order",
        id_col="oid",
        data=pd.DataFrame({"oid": ["o1", "o2"], "express": [True, False]}),
    )
    data.add_event(
        "place_order",
        id_col="eid",
        time_col="ts",
        data=pd.DataFrame({"eid": ["e1", "e2"], "ts": [_DAY, _DAY]}),
    )
    data.add_event_to_object_relation(
        "concerns",
        from_id_col="eid",
        to_id_col="oid",
        to_type=["order"],
        data=pd.DataFrame({"eid": ["e1", "e2"], "oid": ["o1", "o2"]}),
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
