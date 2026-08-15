"""Running a reasoner over a built ontology.

Reasoning is a separate step from building. It reads an `.owl` file through
the Java OWL API, which needs a JVM, and can either report whether the file is
consistent or write back everything the reasoner entails.
"""

import logging
from importlib import resources
from pathlib import Path

import jpype
from owlapy.owl_axiom import OWLObjectPropertyAssertionAxiom
from owlapy.owl_ontology import SyncOntology
from owlapy.owl_reasoner import SyncReasoner

from semanticlog.core.enums import REASONER_BACKENDS as REASONER_BACKENDS
from semanticlog.core.enums import OWLProfile
from semanticlog.core.errors import ConfigError, DependencyError, InputError

logger = logging.getLogger(__name__)

_GENERATORS_ALL = [
    "InferredClassAssertionAxiomGenerator",
    "InferredSubClassAxiomGenerator",
    "InferredDisjointClassesAxiomGenerator",
    "InferredEquivalentClassAxiomGenerator",
    "InferredEquivalentDataPropertiesAxiomGenerator",
    "InferredEquivalentObjectPropertyAxiomGenerator",
    "InferredInverseObjectPropertiesAxiomGenerator",
    "InferredSubDataPropertyAxiomGenerator",
    "InferredSubObjectPropertyAxiomGenerator",
]

_GENERATORS_DL_ONLY = [
    "InferredDataPropertyCharacteristicAxiomGenerator",
    "InferredObjectPropertyCharacteristicAxiomGenerator",
]

# The OWL API generator for entailed property assertions. owlapy leaves it out
# of the mapping `infer_axioms` looks names up in, so it is registered there
# before use.
_PROPERTY_ASSERTIONS = "InferredPropertyAssertionGenerator"

# What a backend answers with UnsupportedOperationException instead of a result,
# found by asking each backend for each generator over the example ontology.
# HermiT and Pellet answer all of them.
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
# warning and a stack trace. The other two quiet Java 24 and newer, which an
# older JVM does not recognize and `ignoreUnrecognized` lets it skip.
_JVM_OPTIONS: tuple[str, ...] = (
    "-Dorg.slf4j.simpleLogger.defaultLogLevel=error",
    "--enable-native-access=ALL-UNNAMED",
    "--sun-misc-unsafe-memory-access=allow",
)


def start_jvm() -> None:
    """Start the JVM the OWL API runs on, unless one is already running.

    owlapy starts one without options the first time an ontology is loaded, so
    this runs first and settles them.

    Raises:
        ValueError: If there is no JVM to start.
    """
    if jpype.isJVMStarted():
        return
    try:
        jpype.getDefaultJVMPath()
    except ValueError as exc:
        raise DependencyError(
            "no JVM found. Reasoning runs the OWL API on Java, so it needs a "
            "JRE or JDK 17 or newer on PATH or named by JAVA_HOME."
        ) from exc

    jars = resources.files("owlapy") / "jar_dependencies"
    classpath = [str(jar) for jar in jars.iterdir() if jar.name.endswith(".jar")]
    jpype.startJVM(*_JVM_OPTIONS, classpath=classpath, ignoreUnrecognized=True)


class Reasoner:
    """A reasoner bound to one ontology file.

    Attributes:
        path: File the ontology was loaded from.
        backend: Name of the reasoner doing the work.
        ontology: The loaded ontology.
        reasoner: The backend doing the work.
        possible_generators: Which sorts of entailment to ask for.
        entails_property_assertions: Whether this backend answers for the
            relations between individuals.
    """

    def __init__(
        self,
        path: str,
        backend: str = "HermiT",
        profile: OWLProfile = OWLProfile.OWL2DL,
    ):
        """Load an ontology and prepare a reasoner over it.

        Args:
            path: Ontology file to read.
            backend: One of HermiT, Pellet or ELK.
            profile: Profile the ontology was built for. Under OWL 2 EL the
                property characteristic entailments are not asked for, since
                that profile has no axioms for them.

        Raises:
            ValueError: If the backend is not one of the three supported, if
                there is no JVM to run the OWL API on, or if the backend
                refuses the ontology, such as HermiT on a timestamp carrying
                microseconds.
            FileNotFoundError: If there is no ontology at `path`. The OWL API
                reports a missing file as a Java exception, which reaches the
                caller as a stack trace rather than as a message.
        """
        if backend not in REASONER_BACKENDS:
            raise ConfigError(f"Unsupported reasoner backend: {backend}")
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"ontology file not found: {path}. 'semlog reason' reads an "
                "ontology that has already been built."
            )
        start_jvm()

        self.path = path
        self.backend = backend
        try:
            self.ontology = SyncOntology(self.path)
            self.reasoner = SyncReasoner(self.ontology, reasoner=backend)
        except Exception as exc:
            others = ", ".join(b for b in REASONER_BACKENDS if b != backend)
            raise InputError(
                f"{backend} will not read {path}:\n"
                f"  {exc}\n"
                f"Try one of the other backends: {others}."
            ) from exc

        unsupported = _UNSUPPORTED.get(backend, frozenset())
        possible = list(_GENERATORS_ALL)
        if profile != OWLProfile.OWL2EL:
            possible.extend(_GENERATORS_DL_ONLY)
        self.possible_generators = [g for g in possible if g not in unsupported]
        self.entails_property_assertions = _PROPERTY_ASSERTIONS not in unsupported

    def materialize(
        self, overwrite_old: bool = True, property_assertions: bool = True
    ) -> None:
        """Add everything the reasoner entails to the ontology.

        Args:
            overwrite_old: Save the result back over the file it came from.
            property_assertions: Also add the entailed relations between
                individuals, so a transitive property such as
                eventually-follows can be read without a reasoner. This is the
                expensive part.
        """
        inferred = list(
            self.reasoner.infer_axioms(inference_types=self.possible_generators)
        )
        if property_assertions and self.entails_property_assertions:
            inferred.extend(self.inferred_property_assertions())
        elif property_assertions:
            logger.info(
                "  %s does not answer for entailed property assertions, "
                "so none were asked for",
                self.backend,
            )
        self.ontology.add_axiom(inferred)
        if overwrite_old:
            self.save()

    def inferred_property_assertions(self) -> list[OWLObjectPropertyAssertionAxiom]:
        """Ask the reasoner for every entailed relation between individuals.

        The OWL API walks the individuals itself, so this is one call into Java
        rather than one per individual and property.

        Returns:
            The entailed assertions. The generator also produces the data
            property ones, which the ontology does not take from here.
        """
        logger.info("  asking %s for entailed property assertions..", self.backend)
        self._register_property_assertions()
        axioms = [
            axiom
            for axiom in self.reasoner.infer_axioms([_PROPERTY_ASSERTIONS])
            if isinstance(axiom, OWLObjectPropertyAssertionAxiom)
        ]
        logger.info(f"  {len(axioms)} entailed object property assertion(s)")
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
        """Report whether the ontology is consistent."""
        return self.reasoner.has_consistent_ontology()

    def save(self, save_path: str | None = None) -> None:
        """Write the ontology out, over the source file unless told otherwise."""
        if save_path is None:
            save_path = self.path
        self.ontology.save(save_path)
