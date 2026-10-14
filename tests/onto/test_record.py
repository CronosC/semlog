import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.onto.record import recorded_annotation, recorded_configuration

FUNCTIONAL = (
    "Ontology(<https://oced>\n"
    r'Annotation(<https://oced#buildConfiguration> "{\"label\": \"a \\\"b\\\"\"}")'
    "\n"
    ")\n"
)
RDF_XML = (
    "<owl:Ontology>\n"
    "  <buildConfiguration>{&quot;label&quot;: &quot;a &amp; b&quot;}"
    "</buildConfiguration>\n"
    "</owl:Ontology>\n"
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [(FUNCTIONAL, {"label": 'a "b"'}), (RDF_XML, {"label": "a & b"})],
    ids=["functional", "rdf_xml"],
)
def test_the_configuration_is_read_back_from_either_syntax(tmp_path, text, expected):
    path = tmp_path / "onto.ofn"
    path.write_text(text, encoding="utf-8")

    assert recorded_configuration(path) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Ontology(<https://oced>)\n",
        'Annotation(<https://oced#buildConfiguration> "not json")\n',
    ],
    ids=["no_annotation", "not_json"],
)
def test_a_file_without_a_readable_annotation_records_nothing(tmp_path, text):
    path = tmp_path / "onto.ofn"
    path.write_text(text, encoding="utf-8")

    assert recorded_annotation(path, vocab.BUILD_CONFIGURATION) is None


def test_a_missing_file_records_nothing(tmp_path):
    assert recorded_annotation(tmp_path / "absent.ofn", vocab.CASE_CONCEPTS) is None


def test_a_configuration_that_is_not_a_mapping_is_not_returned(tmp_path):
    path = tmp_path / "onto.ofn"
    path.write_text(
        'Annotation(<https://oced#buildConfiguration> "[1, 2]")\n', encoding="utf-8"
    )

    assert recorded_configuration(path) is None
