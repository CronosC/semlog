"""An OWL 2 EL build uses nothing that profile forbids."""

from __future__ import annotations

import re

import pytest

from semanticlog.onto.settings import OWLProfile
from tests.onto.scenarios.test_golden import build_lines
from tests.onto.support.configurations import CONFIGURATIONS

_FORBIDDEN_IN_EL = (
    "ObjectUnionOf",
    "ObjectComplementOf",
    "ObjectAllValuesFrom",
    "ObjectMaxCardinality",
    "ObjectExactCardinality",
    "DisjointUnion",
    "InverseObjectProperties",
    "ObjectInverseOf",
    "TransitiveObjectProperty",
    "FunctionalObjectProperty",
    "InverseFunctionalObjectProperty",
    "SymmetricObjectProperty",
    "AsymmetricObjectProperty",
    "DataUnionOf",
    "DataComplementOf",
)

_EL_CONFIGURATIONS = sorted(
    name
    for name, factory in CONFIGURATIONS.items()
    if factory().profile == OWLProfile.OWL2EL
)


def test_the_matrix_actually_covers_el():
    assert _EL_CONFIGURATIONS


@pytest.mark.parametrize("name", _EL_CONFIGURATIONS)
def test_an_el_build_uses_no_el_forbidden_constructor(name, tmp_path):
    lines = build_lines(CONFIGURATIONS[name](), tmp_path)

    offenders = [
        (constructor, line)
        for line in lines
        for constructor in _FORBIDDEN_IN_EL
        if re.search(rf"\b{constructor}\(", line)
    ]
    assert not offenders, (
        f"{name} is an OWL2EL build but emits {len(offenders)} axiom(s) using a "
        "constructor EL does not have:\n"
        + "\n".join(f"  {c}: {line}" for c, line in offenders[:10])
    )


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_a_shared_attribute_keeps_its_domain_wherever_the_profile_allows_one(
    name, tmp_path
):
    settings = CONFIGURATIONS[name]()
    lines = build_lines(settings, tmp_path)
    domains = [line for line in lines if line.startswith("ObjectPropertyDomain(:has_")]

    if settings.profile == OWLProfile.OWL2DL:
        assert any("ObjectUnionOf" in line for line in domains), (
            f"{name}: a DL build should still express a multi-owner domain as a union"
        )
    else:
        assert not any("ObjectUnionOf" in line for line in domains)
