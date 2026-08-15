"""Sink that writes OWL functional syntax straight to a file.

An axiom goes out as it arrives and is not kept, so what a build holds does not
grow with the number of events. It does grow with what the log is made of: one
entry per declared name, and one per distinct attribute value and moment. The
log itself is held in full by the store either way, and for a large log that is
the floor. `benchmarks/streaming_memory.py` measures all three.

Writing the syntax here also puts the literal types under this module's control
instead of the backend's.

The ontology is written under a partial name and moved into place on close,
because the closing bracket is written then and what came before it is not an
ontology.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO

from semanticlog.core.enums import OWLProfile
from semanticlog.onto.sinks.base import (
    _RDFS,
    _XSD,
    OWL_DATATYPES,
    PARTIAL_SUFFIX,
    AxiomSink,
)
from semanticlog.onto.sinks.literals import resolve_literal


class StreamingSink(AxiomSink):
    """Writes each axiom as it arrives, in OWL functional syntax.

    Can be used as a context manager, which closes the file on the way out.

    Attributes:
        path: File being written.
    """

    def __init__(self, path: str, iri: str, profile: OWLProfile):
        """Open the file to write to.

        Writing goes to a partial file beside the destination, which is moved
        into place on close. The header is written when the sink is opened
        rather than here, because the ontology annotations belong inside it
        and they arrive after the sink exists.

        Args:
            path: Where the finished ontology belongs. An existing file there
                is replaced only once this one is complete.
            iri: Base IRI of the ontology.
            profile: Profile of the build this sink serves.
        """
        super().__init__(iri, profile)
        self.path = path
        self._partial = Path(path + PARTIAL_SUFFIX)
        self._file: TextIO | None = open(self._partial, "w", encoding="utf-8")
        self._axiom_count = 0

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

    def write_ontology_header(self, annotations: Mapping[str, str]) -> None:
        """Write the prefixes, open the ontology, and annotate it.

        The annotations go inside `Ontology(...)`, which is what makes them
        annotations on the ontology rather than on an IRI that happens to be
        its name. That is why they all arrive at once and why nothing can be
        added to them afterwards.
        """
        base = self.iri.rstrip("#")
        self._out.write(
            f"Prefix(:=<{self.iri}>)\n"
            f"Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n"
            f"Prefix(rdf:=<http://www.w3.org/1999/02/22-rdf-syntax-ns#>)\n"
            f"Prefix(rdfs:=<{_RDFS}>)\n"
            f"Prefix(xsd:=<{_XSD}>)\n\n"
            f"Ontology(<{base}>\n"
        )
        for property_iri, value in annotations.items():
            self._out.write(f'Annotation(<{property_iri}> "{self._escape(value)}")\n')
            self._axiom_count += 1

    def _axiom(self, text: str) -> None:
        """Write one axiom and count it, opening the ontology if it is not open."""
        self._begin_write()
        self._out.write(text)
        self._out.write("\n")
        self._axiom_count += 1

    @staticmethod
    def _ref(name: str) -> str:
        """Return a name as a prefixed name.

        Nothing is escaped, because a name reaches this having passed
        `vocab.SAFE_LOCAL_NAME`, which allows only characters a prefixed name
        already takes.
        """
        return f":{name}"

    def _class_expression(self, disjuncts: list[str]) -> str:
        """Return one class reference, or their union when there are several."""
        if len(disjuncts) == 1:
            return self._ref(disjuncts[0])
        return "ObjectUnionOf(" + " ".join(self._ref(p) for p in disjuncts) + ")"

    _DECLARATION_KIND = {
        "class": "Class",
        "object": "ObjectProperty",
        "data": "DataProperty",
        "annotation": "AnnotationProperty",
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

    def _literal(self, value: Any) -> str | None:
        """Render a value as a typed literal, or None when it is missing."""
        literal = resolve_literal(value, self.profile)
        if literal is None:
            return None
        return f'"{self._escape(literal.text)}"^^xsd:{literal.datatype}'

    @staticmethod
    def _escape(text: str) -> str:
        """Escape the characters that would end a quoted literal early.

        A quoted string takes only `\\"` and `\\\\`, so a newline stays as it is
        and the axiom carrying it spans more than one line.
        """
        return text.replace("\\", "\\\\").replace('"', '\\"')

    def write_annotations(
        self, annotation: str, pairs: Iterable[tuple[str, str]]
    ) -> None:
        prop = f"rdfs:{annotation}"
        for name, text in pairs:
            self._axiom(
                f"AnnotationAssertion({prop} {self._ref(str(name))} "
                f'"{self._escape(str(text))}")'
            )

    def _finish(self) -> None:
        """Close the ontology and the file, then move it into place."""
        out = self._out
        out.write(")\n")
        out.close()
        self._file = None
        self._partial.replace(self.path)

    def abort(self) -> None:
        """Close the file without finishing the ontology, and remove it.

        What has been written so far is missing its closing bracket, so it is
        not an ontology. It is still under the partial name, so removing it
        leaves whatever was at the destination untouched.
        """
        if self._file is not None:
            self._file.close()
            self._file = None
        self._closed = True
        self._partial.unlink(missing_ok=True)

    def save(self, path: str) -> None:
        """Finish the file this sink has been writing.

        The ontology has been going to disk all along, so this only closes it,
        which is what moves it into place, and it cannot write it somewhere
        else.

        Raises:
            ValueError: If the path is not the one this sink was opened on.
        """
        if path != self.path:
            raise ValueError(
                f"StreamingSink is writing to {self.path!r} and cannot save to "
                f"{path!r}. Open it on the destination path instead."
            )
        self.close()

    def result_target(self) -> tuple[Any, str | None]:
        """Hand back the path, since the ontology went straight to it."""
        return None, self.path
