"""Sink that writes OWL functional syntax straight to a file.

Nothing is kept beyond the name bookkeeping of the base class, so memory stays
roughly flat however many events the log has. Writing the syntax here also puts
the literal types under this module's control instead of the backend's.

The file is only a valid ontology once it has been closed, because the closing
bracket is written then.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any, TextIO

from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks.base import (
    _RDFS,
    _XSD,
    OWL_DATATYPES,
    AxiomSink,
    normalize_value,
)

_PN_LOCAL_ESCAPE = re.compile(r"[~+]")


class StreamingSink(AxiomSink):
    """Writes each axiom as it arrives, in OWL functional syntax.

    Can be used as a context manager, which closes the file on the way out.

    Attributes:
        path: File being written.
    """

    def __init__(self, path: str, iri: str, settings: OntoBuildSettings):
        """Open the file and write the prefixes and ontology header.

        Args:
            path: File to write. An existing file is replaced.
            iri: Base IRI of the ontology.
            settings: Settings of the build this sink serves.
        """
        super().__init__(iri, settings)
        self.path = path
        self._file: TextIO | None = open(path, "w", encoding="utf-8")
        self._axiom_count = 0
        self._write_header()

    @property
    def axiom_count(self) -> int:
        """How many axioms have been written so far."""
        return self._axiom_count

    def __enter__(self) -> StreamingSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def _out(self) -> TextIO:
        """The open file, or a clear failure once it has been closed."""
        if self._file is None:
            raise ValueError(f"StreamingSink for {self.path} is already closed.")
        return self._file

    def _write_header(self) -> None:
        """Write the prefix declarations and open the ontology."""
        base = self.iri.rstrip("#")
        self._out.write(
            f"Prefix(:=<{self.iri}>)\n"
            f"Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n"
            f"Prefix(rdf:=<http://www.w3.org/1999/02/22-rdf-syntax-ns#>)\n"
            f"Prefix(rdfs:=<{_RDFS}>)\n"
            f"Prefix(xsd:=<{_XSD}>)\n\n"
            f"Ontology(<{base}>\n"
        )

    def _axiom(self, text: str) -> None:
        """Write one axiom and count it."""
        self._out.write(text)
        self._out.write("\n")
        self._axiom_count += 1

    @staticmethod
    def _ref(name: str) -> str:
        """Return a prefixed name, escaping the characters the syntax reserves."""
        return f":{_PN_LOCAL_ESCAPE.sub(r'\\\g<0>', name)}"

    def _class_expression(self, disjuncts: list[str]) -> str:
        """Return one class reference, or their union when there are several."""
        if len(disjuncts) == 1:
            return self._ref(disjuncts[0])
        return "ObjectUnionOf(" + " ".join(self._ref(p) for p in disjuncts) + ")"

    _DECLARATION_KIND = {
        "class": "Class",
        "object": "ObjectProperty",
        "data": "DataProperty",
    }

    def write_declaration(self, name: str, kind: str) -> None:
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

    def write_range(self, prop: str, expression: list[str], data_prop: bool) -> None:
        keyword = "DataPropertyRange" if data_prop else "ObjectPropertyRange"
        if len(expression) == 1 and expression[0] in OWL_DATATYPES:
            target = f"xsd:{expression[0]}"
        else:
            target = self._class_expression(expression)
        self._axiom(f"{keyword}({self._ref(prop)} {target})")

    def write_inverse_properties(self, prop: str, inverse: str) -> None:
        self._axiom(f"InverseObjectProperties({self._ref(prop)} {self._ref(inverse)})")

    def write_transitive(self, prop: str) -> None:
        self._axiom(f"TransitiveObjectProperty({self._ref(prop)})")

    def write_class_assertions(
        self, class_name: str, individuals: Iterable[str]
    ) -> None:
        ref = self._ref(class_name)
        for individual in individuals:
            self._axiom(f"ClassAssertion({ref} {self._ref(str(individual))})")

    def write_object_assertions(
        self, prop: str, pairs: Iterable[tuple[str, str]]
    ) -> None:
        ref = self._ref(prop)
        for subject, obj in pairs:
            self._axiom(
                f"ObjectPropertyAssertion({ref} {self._ref(str(subject))} "
                f"{self._ref(str(obj))})"
            )

    def write_data_assertions(
        self, prop: str, pairs: Iterable[tuple[str, Any]]
    ) -> None:
        ref = self._ref(prop)
        for subject, value in pairs:
            literal = self._literal(value)
            if literal is None:
                continue
            self._axiom(
                f"DataPropertyAssertion({ref} {self._ref(str(subject))} {literal})"
            )

    @classmethod
    def _literal(cls, value: Any) -> str | None:
        """Render a value as a typed literal, or None when it is missing."""
        value = normalize_value(value)
        if value is None:
            return None
        if isinstance(value, bool):
            return f'"{str(value).lower()}"^^xsd:boolean'
        if isinstance(value, datetime):
            return f'"{value.isoformat()}"^^xsd:dateTime'
        if isinstance(value, date):
            return f'"{value.isoformat()}"^^xsd:date'
        if isinstance(value, int):
            return f'"{value}"^^xsd:integer'
        if isinstance(value, float):
            return f'"{value}"^^xsd:double'
        return f'"{cls._escape(str(value))}"^^xsd:string'

    @staticmethod
    def _escape(text: str) -> str:
        """Escape the characters that would end a quoted literal early."""
        return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def write_annotations(
        self, annotation: str, pairs: Iterable[tuple[str, str]]
    ) -> None:
        prop = f"rdfs:{annotation}"
        for name, text in pairs:
            self._axiom(
                f"AnnotationAssertion({prop} {self._ref(str(name))} "
                f'"{self._escape(str(text))}")'
            )

    def write_ontology_annotation(self, property_iri: str, value: str) -> None:
        self._axiom(
            f"AnnotationAssertion(<{property_iri}> <{self.iri.rstrip('#')}> "
            f'"{self._escape(value)}")'
        )

    def close(self) -> None:
        """Close the ontology and the file. Closing twice is harmless."""
        if self._file is None:
            return
        self._file.write(")\n")
        self._file.close()
        self._file = None

    def save(self, path: str) -> None:
        """Finish the file this sink has been writing.

        The ontology has been going to disk all along, so this only closes it
        and cannot write it somewhere else.

        Raises:
            ValueError: If the path is not the one this sink was opened on.
        """
        if path != self.path:
            raise ValueError(
                f"StreamingSink is writing to {self.path!r} and cannot save to "
                f"{path!r}. Open it on the destination path instead."
            )
        self.close()
