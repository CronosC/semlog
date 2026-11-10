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
from typing import NoReturn

import jpype
from owlapy.owl_axiom import OWLObjectPropertyAssertionAxiom
from owlapy.owl_ontology import SyncOntology
from owlapy.owl_reasoner import SyncReasoner

from semanticlog.core.enums import OutputFormat, ReasonerBackend
from semanticlog.core.errors import (
    ConfigError,
    DependencyError,
    InputError,
)
from semanticlog.onto.sinks.base import partial_file

logger = logging.getLogger(__name__)

# OWL API document format class of each output format, in
# `org.semanticweb.owlapi.formats`.
_DOCUMENT_FORMATS: dict[OutputFormat, str] = {
    OutputFormat.OFN: "FunctionalSyntaxDocumentFormat",
    OutputFormat.OWL: "RDFXMLDocumentFormat",
    OutputFormat.OWX: "OWLXMLDocumentFormat",
    OutputFormat.TTL: "TurtleDocumentFormat",
    OutputFormat.OMN: "ManchesterSyntaxDocumentFormat",
}

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
_UNSUPPORTED: dict[ReasonerBackend, frozenset[str]] = {
    ReasonerBackend.ELK: frozenset(
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
# warning and a stack trace. The heap share replaces Java's default of a
# quarter of the memory limit and leaves the other half to the Python process.
# The last two silence warnings on Java 24 and newer. An older JVM skips them
# under `ignoreUnrecognized`.
_JVM_OPTIONS: tuple[str, ...] = (
    "-Dorg.slf4j.simpleLogger.defaultLogLevel=error",
    "-XX:MaxRAMPercentage=50",
    "--enable-native-access=ALL-UNNAMED",
    "--sun-misc-unsafe-memory-access=allow",
)


_MIN_JAVA = 17

_INSTALL_JAVA = (
    f"Install a JRE or JDK {_MIN_JAVA} or newer and put it on PATH or in JAVA_HOME."
)

_HEAP_ADVICE = (
    "Java ran out of memory. Raise the Java heap with JAVA_TOOL_OPTIONS=-Xmx<size>, "
    "such as JAVA_TOOL_OPTIONS=-Xmx8g, and run the command again."
)

# How far down the causes of a Java exception to look for an OutOfMemoryError.
_MAX_CAUSES = 20


def start_jvm() -> None:
    """Start the JVM the OWL API runs on, unless one is running, and check its version.

    owlapy starts a JVM without options on the first ontology load. Call this
    before that.

    Raises:
        DependencyError: If there is no JVM, it does not start, its heap is
            too small to start, or it is older than Java 17.
    """
    if not jpype.isJVMStarted():
        try:
            jvm_path = jpype.getDefaultJVMPath()
        except ValueError as exc:
            raise DependencyError(f"No JVM found. {_INSTALL_JAVA}") from exc

        jars = resources.files("owlapy") / "jar_dependencies"
        classpath = [str(jar) for jar in jars.iterdir() if jar.name.endswith(".jar")]
        try:
            jpype.startJVM(*_JVM_OPTIONS, classpath=classpath, ignoreUnrecognized=True)
        except (ValueError, OSError, RuntimeError, SystemError) as exc:
            if "OutOfMemoryError" in str(exc):
                raise DependencyError(_HEAP_ADVICE) from exc
            raise DependencyError(
                f"{_describe_jvm(jvm_path)} does not start: {_one_line(exc)}. "
                f"{_INSTALL_JAVA}"
            ) from exc

    version = jpype.getJVMVersion()
    feature = version[1] if version[0] == 1 else version[0]
    if feature < _MIN_JAVA:
        found = ".".join(str(part) for part in version)
        raise DependencyError(
            f"The OWL API needs Java {_MIN_JAVA} or newer, but the JVM is Java "
            f"{found}. {_INSTALL_JAVA}"
        )


def _describe_jvm(jvm_path: str) -> str:
    """Name a JVM library by the Java version in the `release` file of its JDK, if found."""
    for home in list(Path(jvm_path).parents)[:5]:
        release = home / "release"
        if not release.is_file():
            continue
        for line in release.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("JAVA_VERSION="):
                version = line.partition("=")[2].strip().strip('"')
                return f"Java {version} at {home}"
    return f"The JVM at {jvm_path}"


def _one_line(exc: BaseException) -> str:
    """Return the text of an exception with its whitespace runs collapsed to one space."""
    return " ".join(str(exc).split()).rstrip(".")


def _is_out_of_memory(exc: BaseException) -> bool:
    """Whether a Java OutOfMemoryError raised `exc` or is among its causes."""
    if not jpype.isJVMStarted():
        return False
    throwable = jpype.JClass("java.lang.Throwable")
    out_of_memory = jpype.JClass("java.lang.OutOfMemoryError")
    cause: BaseException | None = exc
    for _ in range(_MAX_CAUSES):
        if cause is None:
            return False
        if isinstance(cause, out_of_memory):
            return True
        cause = cause.getCause() if isinstance(cause, throwable) else cause.__cause__
    return False


def raise_java_error(exc: Exception, message: str) -> NoReturn:
    """Raise the error for an exception from the OWL API or a reasoner, chained to it.

    Args:
        exc: The exception raised.
        message: What failed and what to change, for an exception other than
            a heap exhaustion. The exception text is appended.

    Raises:
        DependencyError: If Java ran out of memory, saying how to raise the
            heap.
        InputError: Otherwise, holding `message`.
    """
    if _is_out_of_memory(exc):
        raise DependencyError(_HEAP_ADVICE) from exc
    raise InputError(f"{message} Java reported: {_one_line(exc)}.") from exc


class Reasoner:
    """A reasoner bound to one ontology file.

    Args:
        path: Ontology file to read.
        backend: One of HermiT, Pellet or ELK.

    Attributes:
        path: File the ontology was loaded from.
        backend: The reasoner backend.
        ontology: The loaded ontology.
        reasoner: The owlapy reasoner.
        possible_generators: Generators this backend supports.
        entails_property_assertions: Whether the backend supports entailed
            property assertions.

    Raises:
        ConfigError: If the backend is not one of the three supported.
        DependencyError: If there is no usable JVM, or Java ran out of memory.
        InputError: If the ontology cannot be loaded, or the backend cannot
            start over it. HermiT rejects a timestamp with microseconds.
        FileNotFoundError: If there is no ontology at `path`.
    """

    def __init__(
        self,
        path: str,
        backend: ReasonerBackend | str = ReasonerBackend.HERMIT,
    ) -> None:
        if backend not in ReasonerBackend:
            raise ConfigError(
                f"Reasoner backend {backend!r} is not one of "
                f"{', '.join(b.value for b in ReasonerBackend)}."
            )
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Ontology file not found: {path}. Build it first with 'semlog build'."
            )
        self.path = path
        self.backend = ReasonerBackend(backend)
        self.ontology = load_ontology(path)
        try:
            self.reasoner = SyncReasoner(self.ontology, reasoner=self.backend.value)
        except Exception as exc:
            raise_java_error(
                exc,
                f"{self.backend.value} cannot start over this ontology. Try one of "
                f"the other backends: {self._others()}.",
            )

        unsupported = _UNSUPPORTED.get(self.backend, frozenset())
        self.possible_generators = [g for g in _GENERATORS if g not in unsupported]
        self.entails_property_assertions = _PROPERTY_ASSERTIONS not in unsupported

    def materialize(self, property_assertions: bool = True) -> int:
        """Add everything the reasoner entails to the ontology.

        Nothing is written. `save` or `overwrite` writes the result.

        Args:
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
                self.backend.value,
            )
        before = self._axiom_count()
        with self._reporting("adding the entailed axioms"):
            self.ontology.add_axiom(inferred)
        return self._axiom_count() - before

    def _axiom_count(self) -> int:
        """Axiom count of the OWL API ontology."""
        return int(self.ontology.get_owlapi_ontology().getAxiomCount())

    def inferred_property_assertions(self) -> list[OWLObjectPropertyAssertionAxiom]:
        """Return every entailed object property assertion.

        Data property assertions from the generator are dropped.

        Raises:
            InputError: If the backend fails.
        """
        logger.info("  asking %s for entailed property assertions", self.backend.value)
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
        from org.semanticweb.owlapi.util import (
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
        """Wrap a backend exception in an error naming the backend and the step.

        Raises:
            DependencyError: If Java ran out of memory.
            InputError: If the backend raised anything else.
        """
        try:
            yield
        except Exception as exc:
            raise_java_error(
                exc,
                f"{self.backend.value} failed while {what} over {self.path}. Try "
                f"one of the other backends: {self._others()}.",
            )

    def _others(self) -> str:
        """Return the names of the other backends, comma-separated."""
        return ", ".join(b.value for b in ReasonerBackend if b is not self.backend)

    def save(self, path: str) -> None:
        """Write the ontology to `path`, in the format its suffix names.

        Raises:
            ConfigError: If the suffix of `path` names no format.
            DependencyError: If Java ran out of memory.
            InputError: If the OWL API cannot write `path`.
        """
        save_ontology(self.ontology, OutputFormat.of_path(path), Path(path))

    def overwrite(self) -> None:
        """Write the ontology back over the source file, in the source format.

        The write goes to a partial file beside the source, which then
        replaces it.

        Raises:
            DependencyError: If Java ran out of memory.
        """
        try:
            with partial_file(self.path) as partial:
                self.ontology.save(str(partial))
        except Exception as exc:
            if _is_out_of_memory(exc):
                raise DependencyError(_HEAP_ADVICE) from exc
            raise


def load_ontology(path: str | Path) -> SyncOntology:
    """Start the JVM and load the ontology in `path` through the OWL API.

    Raises:
        DependencyError: If there is no usable JVM, or Java ran out of memory.
        InputError: If the OWL API cannot read the file.
    """
    start_jvm()
    try:
        return SyncOntology(str(path))
    except Exception as exc:
        raise_java_error(
            exc,
            f"The OWL API cannot read {path}. Build the ontology again, or "
            "check that a build wrote the file.",
        )


def save_ontology(
    ontology: SyncOntology, output_format: OutputFormat, target: Path
) -> None:
    """Write the ontology to `target` in one format.

    The write goes to a partial file beside the target, which then replaces
    it. The prefixes of the source document are kept.

    Raises:
        DependencyError: If Java ran out of memory. No file is left.
        InputError: If the OWL API cannot write the file. No file is left.
    """
    manager = ontology.owlapi_manager
    owlapi_ontology = ontology.get_owlapi_ontology()
    source_format = manager.getOntologyFormat(owlapi_ontology)
    document_format = jpype.JClass(
        f"org.semanticweb.owlapi.formats.{_DOCUMENT_FORMATS[output_format]}"
    )()
    if document_format.isPrefixOWLDocumentFormat() and (
        source_format is not None and source_format.isPrefixOWLDocumentFormat()
    ):
        document_format.copyPrefixesFrom(source_format.asPrefixOWLDocumentFormat())
    try:
        with partial_file(target) as partial:
            iri = jpype.JClass("org.semanticweb.owlapi.model.IRI").create(
                jpype.JClass("java.io.File")(str(partial))
            )
            manager.saveOntology(owlapi_ontology, document_format, iri)
    except Exception as exc:
        raise_java_error(
            exc,
            f"Cannot write {target} as {output_format.value}. Check that the "
            "directory is writable and has space.",
        )
