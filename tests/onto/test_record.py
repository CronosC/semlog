"""Reading the build record back out of an ontology file."""

import pytest

from semlog.onto.record import BuildRecord, read_record

CASES = r'Annotation(<https://oced#caseConcepts> "[\"order\"]")' "\n"
FUNCTIONAL = (
    "Ontology(<https://oced>\n"
    r'Annotation(<https://oced#buildConfiguration> "{\"label\": \"a \\\"b\\\"\"}")'
    "\n"
    f"{CASES}"
    ")\n"
)
OWL_API = (
    "Prefix(oced:=<https://oced#>)\n"
    "Ontology(<https://oced>\n"
    r'Annotation(oced:buildConfiguration "{\"label\": \"a \\\"b\\\"\"}")'
    "\n"
    r'Annotation(oced:caseConcepts "[\"order\"]")'
    "\n"
    ")\n"
)
RDF_XML = (
    "<owl:Ontology>\n"
    "  <buildConfiguration>{&quot;label&quot;: &quot;a &amp; b&quot;}"
    "</buildConfiguration>\n"
    "  <caseConcepts>[&quot;order&quot;]</caseConcepts>\n"
    "</owl:Ontology>\n"
)


@pytest.mark.parametrize(
    ("text", "label"),
    [(FUNCTIONAL, 'a "b"'), (OWL_API, 'a "b"'), (RDF_XML, "a & b")],
    ids=["functional", "functional_prefixed", "rdf_xml"],
)
def test_the_record_is_read_back_from_any_writer(tmp_path, text, label):
    path = tmp_path / "onto.ofn"
    path.write_text(text, encoding="utf-8")

    assert read_record(path) == BuildRecord({"label": label}, ["order"])


@pytest.mark.parametrize(
    "text",
    [
        "Ontology(<https://oced>)\n",
        f'Annotation(<https://oced#buildConfiguration> "not json")\n{CASES}',
        f'Annotation(<https://oced#buildConfiguration> "[1, 2]")\n{CASES}',
        'Annotation(<https://oced#buildConfiguration> "{}")\n',
    ],
    ids=["no_annotation", "not_json", "not_a_mapping", "no_case_concepts"],
)
def test_a_header_without_a_complete_record_records_nothing(tmp_path, text):
    path = tmp_path / "onto.ofn"
    path.write_text(text, encoding="utf-8")

    assert read_record(path) is None


def test_a_missing_file_records_nothing(tmp_path):
    assert read_record(tmp_path / "absent.ofn") is None
