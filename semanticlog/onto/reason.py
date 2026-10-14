"""Running a reasoner over a built ontology file.

`Reasoner` reads the file through the Java OWL API, which needs a JVM. It
reports whether the ontology is consistent, or adds what the reasoner entails
and writes the file back.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

import jpype
from owlapy.owl_axiom import OWLObjectPropertyAssertionAxiom
from owlapy.owl_ontology import SyncOntology
from owlapy.owl_reasoner import SyncReasoner

from semanticlog.core.enums import REASONER_BACKENDS as REASONER_BACKENDS
from semanticlog.core.errors import (
    ConfigError,
    DependencyError,
    InputError,
    SemlogError,
)
from semanticlog.onto.sinks.base import PARTIAL_SUFFIX

logger = logging.getLogger(__name__)

_GENERATORS = [
    "InferredClassAssertionAxiomGenerator",
    "InferredSubClassAxiomGenerator",
    "InferredDisjointClassesAxiomGenerator",
    "InferredEquivalentClassAxiomGenerator",
    "InferredEquivalentDataPropertiesAxiomGenerator",
    "InferredEquivalentObjectPropertyAxiomGenerator",
    "InferredInverseObjectPropertiesAxiomGenerator",
    "InferredSubDataPropertyAxiomGenerator",
    "InferredSubObjectPropertyAxiomGenerator",
    "InferredDataPropertyCharacteristicAxiomGenerator",
    "InferredObjectPropertyCharacteristicAxiomGenerator",
]

# The OWL API generator for entailed property assertions. owlapy leaves it out
# of the mapping `infer_axioms` looks names up in, so it is registered there
# before use.
_PROPERTY_ASSERTIONS = "InferredPropertyAssertionGenerator"

# Generators a backend raises UnsupportedOperationException for, found by
# running each generator on each backend over the example ontology. HermiT and
# Pellet support all of them.
_UNSUPPORTED: dict[str, frozenset[str]] = {
    "ELK": frozenset(
        {
            "InferredDisjointClassesAxiomGenerator",
            "InferredEquivalentDataPropertiesAxiomGenerator",
            "InferredInverseObjectPropertiesAxiomGenerator",
            "InferredSubDataPropertyAxiomGenerator",
            _PROPERTY_ASSERTIONS,
        }
    ),
}

# Options for the JVM the OWL API runs on. The slf4j level silences the parser,
# which reports the RDF/XML attempt it makes on a functional syntax file as a
# warning and a stack trace. The other two silence warnings on Java 24 and
# newer. An older JVM skips them under `ignoreUnrecognized`.
_JVM_OPTIONS: tuple[str, ...] = (
    "-Dorg.slf4j.simpleLogger.defaultLogLevel=error",
    "--enable-native-access=ALL-UNNAMED",
    "--sun-misc-unsafe-memory-access=allow",
)


def start_jvm() -> None:
    """Start the JVM the OWL API runs on, unless one is running.

    owlapy starts a JVM without options on the first ontology load. Call this
    before that.

    Raises:
        DependencyError: If there is no JVM to start.
    """
    if jpype.isJVMStarted():
        return
    try:
        jpype.getDefaultJVMPath()
    except ValueError as exc:
        raise DependencyError(
            "No JVM found. Install a JRE or JDK 17 or newer and put it on PATH "
            "or in JAVA_HOME."
        ) from exc

    jars = resources.files("owlapy") / "jar_dependencies"
    classpath = [str(jar) for jar in jars.iterdir() if jar.name.endswith(".jar")]
    jpype.startJVM(*_JVM_OPTIONS, classpath=classpath, ignoreUnrecognized=True)


class Reasoner:
    """A reasoner bound to one ontology file.

    Args:
        path: Ontology file to read.
        backend: One of HermiT, Pellet or ELK.

    Attributes:
        path: File the ontology was loaded from.
        backend: Reasoner name.
        ontology: The loaded ontology.
        reasoner: The owlapy reasoner.
        possible_generators: Generators this backend supports.
        entails_property_assertions: Whether the backend supports entailed
            property assertions.

    Raises:
        ConfigError: If the backend is not one of the three supported.
        DependencyError: If there is no JVM to run the OWL API on.
        InputError: If the ontology cannot be loaded, or the backend cannot
            start over it. HermiT rejects a timestamp with microseconds.
        FileNotFoundError: If there is no ontology at `path`.
    """

    def __init__(
        self,
        path: str,
        backend: str = "HermiT",
    ) -> None:
        if backend not in REASONER_BACKENDS:
            raise ConfigError(
                f"Reasoner backend {backend!r} is not one of "
                f"{', '.join(REASONER_BACKENDS)}."
            )
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Ontology file not found: {path}. Build it first with 'semlog build'."
            )
        start_jvm()

        self.path = path
        self.backend = backend
        others = ", ".join(b for b in REASONER_BACKENDS if b != backend)
        try:
            self.ontology = SyncOntology(self.path)
        except Exception as exc:
            raise InputError(
                f"The OWL API cannot read {path}:\n  {exc}\n"
                "Build the ontology again, or check that a build wrote the file."
            ) from exc
        try:
            self.reasoner = SyncReasoner(self.ontology, reasoner=backend)
        except Exception as exc:
            raise InputError(
                f"{backend} cannot start over this ontology:\n  {exc}\n"
                f"Try one of the other backends: {others}."
            ) from exc

        unsupported = _UNSUPPORTED.get(backend, frozenset())
        self.possible_generators = [g for g in _GENERATORS if g not in unsupported]
        self.entails_property_assertions = _PROPERTY_ASSERTIONS not in unsupported

    def materialize(
        self, overwrite_old: bool = True, property_assertions: bool = True
    ) -> int:
        """Add everything the reasoner entails to the ontology.

        Args:
            overwrite_old: Save the result back over the file it came from.
            property_assertions: Also add entailed object property assertions,
                such as the closure of a transitive property. This is the slow
                step.

        Returns:
            How many axioms were added. Entailments the ontology holds are not
            counted.

        Raises:
            InputError: If the backend fails while reasoning.
        """
        with self._reporting("asking for entailments"):
            inferred = list(
                self.reasoner.infer_axioms(inference_types=self.possible_generators)
            )
        if property_assertions and self.entails_property_assertions:
            inferred.extend(self.inferred_property_assertions())
        elif property_assertions:
            logger.info(
                "  %s does not support entailed property assertions, skipped",
                self.backend,
            )
        before = self._axiom_count()
        with self._reporting("adding the entailed axioms"):
            self.ontology.add_axiom(inferred)
        added = self._axiom_count() - before
        if overwrite_old:
            self.save()
        return added

    def _axiom_count(self) -> int:
        """Axiom count of the OWL API ontology."""
        return int(self.ontology.get_owlapi_ontology().getAxiomCount())

    def inferred_property_assertions(self) -> list[OWLObjectPropertyAssertionAxiom]:
        """Return every entailed object property assertion.

        Data property assertions from the generator are dropped.

        Raises:
            InputError: If the backend fails.
        """
        logger.info("  asking %s for entailed property assertions..", self.backend)
        self._register_property_assertions()
        with self._reporting("asking for entailed property assertions"):
            axioms = [
                axiom
                for axiom in self.reasoner.infer_axioms([_PROPERTY_ASSERTIONS])
                if isinstance(axiom, OWLObjectPropertyAssertionAxiom)
            ]
        logger.info("  %d entailed object property assertion(s) returned", len(axioms))
        return axioms

    def _register_property_assertions(self) -> None:
        """Put the missing generator in the mapping `infer_axioms` reads."""
        if _PROPERTY_ASSERTIONS in self.reasoner.inference_types_mapping:
            return
        from org.semanticweb.owlapi.util import (  # noqa: PLC0415
            InferredPropertyAssertionGenerator,
        )

        self.reasoner.inference_types_mapping[_PROPERTY_ASSERTIONS] = (
            InferredPropertyAssertionGenerator()
        )

    def check(self) -> bool:
        """Return whether the ontology is consistent.

        Raises:
            InputError: If the backend fails. An inconsistent ontology returns
                False.
        """
        with self._reporting("checking consistency"):
            return bool(self.reasoner.has_consistent_ontology())

    @contextmanager
    def _reporting(self, what: str) -> Iterator[None]:
        """Wrap a backend exception in an InputError naming the backend and the step.

        Raises:
            InputError: If the backend raised.
        """
        try:
            yield
        except (SemlogError, FileNotFoundError):
            raise
        except Exception as exc:
            others = ", ".join(b for b in REASONER_BACKENDS if b != self.backend)
            raise InputError(
                f"{self.backend} failed while {what} over {self.path}:\n"
                f"  {exc}\n"
                f"Try one of the other backends: {others}."
            ) from exc

    def save(self, save_path: str | None = None) -> None:
        """Write the ontology to `save_path`, or over the source file.

        The write goes to a partial file beside the target, which then
        replaces it.
        """
        target = save_path if save_path is not None else self.path
        partial = Path(target + PARTIAL_SUFFIX)
        try:
            self.ontology.save(str(partial))
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
