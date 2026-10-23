"""Sink that writes OWL functional syntax to a file.

`StreamingSink` writes each axiom as it arrives and keeps none. Its memory
grows with the number of declared names, distinct attribute values and
timestamps. The store holds the whole log under either sink.

The file is written under `PARTIAL_SUFFIX` and moved to the destination on
close.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from types import TracebackType
from typing import TextIO

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import IndividualKind, OWLProfile, SinkEntity
from semanticlog.core.vocabulary import format_number
from semanticlog.onto.sinks.base import PARTIAL_SUFFIX, AxiomSink
from semanticlog.onto.sinks.literals import RDFS, XSD, Literal


class StreamingSink(AxiomSink):
    """Writes each axiom as it arrives, in OWL functional syntax.

    Usable as a context manager. Exiting closes the sink, or aborts it when
    the block raised.

    Args:
        path: Destination file. Replaced once the ontology is complete.
        iri: Base IRI of the ontology.
        profile: Profile of the build.
        tbox_iri: Namespace of the schema names.

    Attributes:
        path: Destination file. Replaced once the ontology is complete.
    """

    def __init__(
        self,
        path: str,
        iri: str,
        profile: OWLProfile,
        tbox_iri: str = vocab.TBOX_IRI,
    ) -> None:
        super().__init__(iri, profile, tbox_iri)
        self.path = path
        self._partial = Path(path + PARTIAL_SUFFIX)
        self._file: TextIO | None = open(self._partial, "w", encoding="utf-8")

    def __enter__(self) -> StreamingSink:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is None:
            self.close()
        else:
            self.abort()

    @property
    def _out(self) -> TextIO:
        """The open file. Raises ValueError after close."""
        if self._file is None:
            raise ValueError(f"StreamingSink for {self.path} is closed.")
        return self._file

    def write_ontology_header(self, annotations: Mapping[str, str]) -> None:
        """Write the prefixes, the `Ontology(` line and its annotations.

        Functional syntax places ontology annotations inside `Ontology(...)`,
        before the axioms.
        """
        self._out.write(f"Prefix(:=<{self.iri}>)\n")
        for kind in IndividualKind:
            self._out.write(f"Prefix({kind.value}:=<{self.iri}{kind.path}>)\n")
        self._out.write(
            f"Prefix({vocab.TBOX_PREFIX}:=<{self.tbox_iri}>)\n"
            f"Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n"
            f"Prefix(rdf:=<http://www.w3.org/1999/02/22-rdf-syntax-ns#>)\n"
            f"Prefix(rdfs:=<{RDFS}>)\n"
            f"Prefix(xsd:=<{XSD}>)\n\n"
            f"Ontology(<{self.ontology_iri}>\n"
        )
        for property_iri, value in annotations.items():
            self._out.write(f'Annotation(<{property_iri}> "{self._escape(value)}")\n')
            self._axiom_count += 1

    def _axiom(self, text: str) -> None:
        """Write one axiom and count it, opening the ontology if it is not open."""
        self._begin_write()
        self._out.write(text)
        self._out.write("\n")

    @staticmethod
    def _ref(name: str, kind: IndividualKind | None = None) -> str:
        """Return a name as a prefixed name.

        A `Term` goes under the TBox prefix, an individual under the prefix of
        its kind, and any other name under the log prefix. Names have matched
        `vocab.SAFE_LOCAL_NAME`, so nothing is escaped.
        """
        if vocab.is_tbox(name):
            return f"{vocab.TBOX_PREFIX}:{name}"
        if kind is None:
            return f":{name}"
        return f"{kind.value}:{name}"

    def _class_expression(self, disjuncts: list[str]) -> str:
        """Return one class reference, or their union when there are several."""
        if len(disjuncts) == 1:
            return self._ref(disjuncts[0])
        return "ObjectUnionOf(" + " ".join(self._ref(p) for p in disjuncts) + ")"

    _DECLARATION_KIND: dict[SinkEntity, str] = {
        SinkEntity.CLASS: "Class",
        SinkEntity.OBJECT_PROPERTY: "ObjectProperty",
        SinkEntity.DATA_PROPERTY: "DataProperty",
        SinkEntity.ANNOTATION_PROPERTY: "AnnotationProperty",
    }

    def write_declaration(self, name: str, kind: SinkEntity) -> None:
        self._axiom(f"Declaration({self._DECLARATION_KIND[kind]}({self._ref(name)}))")

    def write_sub_class_of(self, sub: str, superclass: list[str]) -> None:
        self._axiom(
            f"SubClassOf({self._ref(sub)} {self._class_expression(superclass)})"
        )

    def write_sub_property_of(self, sub: str, parent: str, data_prop: bool) -> None:
        keyword = "SubDataPropertyOf" if data_prop else "SubObjectPropertyOf"
        self._axiom(f"{keyword}({self._ref(sub)} {self._ref(parent)})")

    def write_domain(self, prop: str, expression: list[str], data_prop: bool) -> None:
        keyword = "DataPropertyDomain" if data_prop else "ObjectPropertyDomain"
        self._axiom(
            f"{keyword}({self._ref(prop)} {self._class_expression(expression)})"
        )

    def write_range(self, prop: str, expression: list[str]) -> None:
        self._axiom(
            f"ObjectPropertyRange({self._ref(prop)} "
            f"{self._class_expression(expression)})"
        )

    def write_data_range(self, prop: str, datatype: str) -> None:
        self._axiom(f"DataPropertyRange({self._ref(prop)} xsd:{datatype})")

    def write_inverse_properties(self, prop: str, inverse: str) -> None:
        self._axiom(f"InverseObjectProperties({self._ref(prop)} {self._ref(inverse)})")

    def write_transitive(self, prop: str) -> None:
        self._axiom(f"TransitiveObjectProperty({self._ref(prop)})")

    def write_bin_definition(
        self,
        bin_class: str,
        attribute_class: str,
        value_prop: str,
        lower: float | None,
        upper: float | None,
    ) -> None:
        facets = []
        if lower is not None:
            facets.append(f'xsd:minExclusive "{format_number(lower)}"^^xsd:decimal')
        if upper is not None:
            facets.append(f'xsd:maxInclusive "{format_number(upper)}"^^xsd:decimal')
        in_range = (
            f"DataSomeValuesFrom({self._ref(value_prop)} "
            f"DatatypeRestriction(xsd:decimal {' '.join(facets)}))"
        )
        self._axiom(
            f"EquivalentClasses({self._ref(bin_class)} "
            f"ObjectIntersectionOf({self._ref(attribute_class)} {in_range}))"
        )

    def write_functional(self, prop: str, data_prop: bool) -> None:
        keyword = "FunctionalDataProperty" if data_prop else "FunctionalObjectProperty"
        self._axiom(f"{keyword}({self._ref(prop)})")

    def write_inverse_functional(self, prop: str) -> None:
        self._axiom(f"InverseFunctionalObjectProperty({self._ref(prop)})")

    def write_disjoint_classes(self, classes: list[str]) -> None:
        self._axiom(f"DisjointClasses({' '.join(self._ref(c) for c in classes)})")

    def write_class_assertions(
        self, class_name: str, individuals: Iterable[str], kind: IndividualKind
    ) -> None:
        ref = self._ref(class_name)
        for individual in individuals:
            self._axiom(f"ClassAssertion({ref} {self._ref(str(individual), kind)})")

    def write_object_assertions(
        self,
        prop: str,
        pairs: Iterable[tuple[str, str]],
        kinds: tuple[IndividualKind, IndividualKind],
    ) -> None:
        ref = self._ref(prop)
        subject_kind, target_kind = kinds
        for subject, obj in pairs:
            self._axiom(
                f"ObjectPropertyAssertion({ref} {self._ref(str(subject), subject_kind)} "
                f"{self._ref(str(obj), target_kind)})"
            )

    def write_data_assertions(
        self, prop: str, pairs: Iterable[tuple[str, Literal]], kind: IndividualKind
    ) -> None:
        ref = self._ref(prop)
        for subject, literal in pairs:
            text = f'"{self._escape(literal.text)}"^^xsd:{literal.datatype}'
            self._axiom(
                f"DataPropertyAssertion({ref} {self._ref(str(subject), kind)} {text})"
            )

    @staticmethod
    def _escape(text: str) -> str:
        """Escape backslash and double quote.

        A newline is not escaped, so an axiom may span lines.
        """
        return text.replace("\\", "\\\\").replace('"', '\\"')

    def write_annotations(
        self,
        annotation: str,
        pairs: Iterable[tuple[str, str]],
        kind: IndividualKind | None = None,
    ) -> None:
        prop = f"rdfs:{annotation}"
        for name, text in pairs:
            self._axiom(
                f"AnnotationAssertion({prop} {self._ref(name, kind)} "
                f'"{self._escape(str(text))}")'
            )

    def write_entity_annotation(
        self, prop: str, name: str, value: str, iri: bool
    ) -> None:
        target = self._ref(value) if iri else f'"{self._escape(value)}"'
        self._axiom(
            f"AnnotationAssertion({self._ref(prop)} {self._ref(name)} {target})"
        )

    def _finish(self) -> None:
        """Close the ontology and the file, then move it into place."""
        out = self._out
        out.write(")\n")
        out.close()
        self._file = None
        self._partial.replace(self.path)

    def abort(self) -> None:
        """Close and remove the partial file. The destination is untouched."""
        if self._file is not None:
            self._file.close()
            self._file = None
        self._closed = True
        self._partial.unlink(missing_ok=True)

    def save(self, path: str) -> None:
        """Close the sink. `path` must be the path the sink was opened on.

        Raises:
            ValueError: If the path is not the one this sink was opened on.
        """
        if path != self.path:
            raise ValueError(
                f"StreamingSink is writing to {self.path!r} and cannot save to "
                f"{path!r}. Open the sink on the destination path."
            )
        self.close()

    def result_target(self) -> tuple[None, str]:
        """Return no ontology and the path written."""
        return None, self.path
