"""The functional-syntax reader in `tests/support/functional.py`, on files the OWL API rewrote."""

from pathlib import Path

from tests.support.emitted import Emitted
from tests.support.functional import read_functional

# An excerpt of a build that `reason` materialized. The OWL API writes a name
# whose local part starts with a digit as a full IRI, and adds comment lines.
_MATERIALIZED = """\
Prefix(:=<https://example.org/oced/log/>)
Prefix(oced:=<https://example.org/oced#>)
Prefix(event:=<https://example.org/oced/log/event/>)
Prefix(time:=<https://example.org/oced/log/time/>)


Ontology(<https://example.org/oced/log>

############################
#   Named Individuals
############################

# Individual: <https://example.org/oced/log/time/20240103T000000_000000> (<https://example.org/oced/log/time/20240103T000000_000000>)

Declaration(NamedIndividual(<https://example.org/oced/log/time/20240103T000000_000000>))
Declaration(Class(oced:Time))
ClassAssertion(oced:Time <https://example.org/oced/log/time/20240103T000000_000000>)
DataPropertyAssertion(oced:time_value <https://example.org/oced/log/time/20240103T000000_000000> "2024-01-03T00:00:00"^^xsd:dateTime)
ObjectPropertyAssertion(oced:observed_at event:pay_e1 <https://example.org/oced/log/time/20240103T000000_000000>)

############################
#   Classes
############################

)
"""


def _read(tmp_path: Path) -> Emitted:
    path = tmp_path / "materialized.ofn"
    path.write_text(_MATERIALIZED, encoding="utf-8")
    return read_functional(path)


def test_an_individual_written_as_a_full_iri_is_read_by_its_local_name(tmp_path):
    emitted = _read(tmp_path)

    assert emitted.types["20240103T000000_000000"] == {"Time"}
    assert ("observed_at", "pay_e1", "20240103T000000_000000") in emitted.relations
    assert ("time_value", "20240103T000000_000000") in emitted.data_assertions


def test_a_comment_line_is_not_read_as_an_axiom(tmp_path):
    emitted = _read(tmp_path)

    assert not [line for line in emitted.axiom_lines if line.startswith("#")]
    assert emitted.duplicate_axioms == {}
