"""Nothing is used before the build has declared it."""

from __future__ import annotations

import pytest

from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import DEFAULT_IRI
from tests.onto.support.configurations import CONFIGURATIONS, build_store
from tests.onto.support.recording import RecordingSink


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_no_individual_is_written_before_its_name_is_declared(name):
    settings = CONFIGURATIONS[name]()
    sink = RecordingSink(settings)

    OCEDOntoBuilder(DEFAULT_IRI, settings, sink=sink).build_ontology(build_store())

    violations = sink.declaration_order_violations()
    assert not violations, (
        f"{name}: {len(violations)} name(s) were asserted as an individual and "
        "only declared as a class or property afterwards, which the sink's name "
        "ledger cannot detect:\n"
        + "\n".join(f"  {n!r} asserted into {cls!r}" for n, cls in violations)
    )


def test_the_check_would_notice_a_violation():
    settings = CONFIGURATIONS["dl_df"]()
    sink = RecordingSink(settings)

    sink.assert_individual("late", "some_class")
    sink.add_class("late", source="a class declared too late")

    assert sink.declaration_order_violations() == [("late", "some_class")]
