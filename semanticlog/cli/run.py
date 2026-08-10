"""The pipelines behind each subcommand, with no argparse in sight.

Every function here takes plain values or a :class:`~semanticlog.cli.config.RunConfig`
and returns a result object, so the same steps can be driven from a test, a
notebook, or the command line.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semanticlog.cli.config import RunConfig
from semanticlog.cli.logs import FileLog
from semanticlog.cli.manifest import RunManifest
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.loading.pm4pyloader import PM4PYLoader
from semanticlog.data.oced import OCEDStore, RelationKind
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.semlog import SemanticLog
from semanticlog.onto.settings import DEFAULT_IRI, OcelExtension, OwlProfile

logger = logging.getLogger("semlog.run")

_MB = 1024 * 1024


@contextmanager
def _timed(timings: dict[str, float], stage: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = time.perf_counter() - started


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


@dataclass
class BuildReport:
    """What a ``semlog build`` invocation did, per input file."""

    built: list[Path] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    manifests: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        parts = [f"{len(self.built)} built"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts)


def load_and_clean(path: Path, extensions: Sequence[OcelExtension]) -> OCEDStore:
    """Read an OCEL file and sanitise every name so it is safe in an IRI."""
    store = PM4PYLoader(str(path), {}, ocel_extensions=list(extensions)).get_oced()
    return clean_oced_data(store)


def choose_case_concepts(cfg: RunConfig, store: OCEDStore) -> tuple[list[str], str]:
    """Return the case concepts for one input and where they came from.

    The source string ends up in the manifest, because auto-selection is a
    modelling decision and a reader of the result needs to know it was made.
    """
    if cfg.case_concepts is not None:
        return list(cfg.case_concepts), "config"
    chosen = store.get_most_common_objects(n=cfg.auto_case_concepts)
    return chosen, f"auto(n={cfg.auto_case_concepts})"


def _relation_rows(store: OCEDStore, kind: RelationKind) -> int:
    return sum(
        len(store.get_relation(name).data) for name in store.get_relation_types(kind)
    )


def _store_stats(store: OCEDStore) -> dict[str, int]:
    return {
        "activities": len(store.get_activities()),
        "events": sum(len(store.get_events(a).data) for a in store.get_activities()),
        "object_types": len(store.get_object_types()),
        "objects": sum(len(table.data) for table in store.get_objects().values()),
        "event_to_object_relations": _relation_rows(store, RelationKind.E2O),
        "object_to_object_relations": _relation_rows(store, RelationKind.O2O),
    }


def _ontology_stats(semlog: SemanticLog) -> dict[str, int]:
    """Signature sizes of the built ontology.

    Signature counts rather than axiom counts: owlapy's in-memory ``Ontology``
    raises ``NotImplementedError`` from ``get_tbox_axioms``/``get_abox_axioms``.
    """
    onto = semlog.require_onto()
    return {
        "classes": sum(1 for _ in onto.classes_in_signature()),
        "object_properties": sum(1 for _ in onto.object_properties_in_signature()),
        "data_properties": sum(1 for _ in onto.data_properties_in_signature()),
        "individuals": sum(1 for _ in onto.individuals_in_signature()),
    }


def build_one(cfg: RunConfig, input_path: Path) -> dict[str, Any]:
    """Build one ontology and its manifest, returning the manifest contents."""
    onto_path = cfg.ontology_path(input_path)
    timings: dict[str, float] = {}

    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {onto_path}")

    logger.info("Loading OCEL data..")
    with _timed(timings, "load"):
        store = load_and_clean(input_path, cfg.settings.ocel_extensions)

    case_concepts, source = choose_case_concepts(cfg, store)
    logger.info(f"Case concepts ({source}): {case_concepts or 'none'}")
    settings = cfg.settings_for(case_concepts)

    logger.info("Building ontology..")
    with _timed(timings, "build"):
        semlog = OCEDOntoBuilder(DEFAULT_IRI, settings).build_ontology(
            store, source=str(input_path)
        )

    with _timed(timings, "write"):
        onto_path.parent.mkdir(parents=True, exist_ok=True)
        semlog.write_ontology(str(onto_path))
    logger.info(f"Ontology saved: {onto_path}")

    outputs: dict[str, str | None] = {
        "ontology": str(onto_path),
        "manifest": str(cfg.manifest_path(input_path)),
        "backmap": None,
        "store": None,
    }
    if cfg.save_backmap:
        backmap_path = cfg.backmap_path(input_path)
        store.save_backmap(backmap_path)
        outputs["backmap"] = str(backmap_path)
        logger.info(f"Name backmap saved: {backmap_path}")
    if cfg.save_store:
        store_path = cfg.store_path(input_path)
        store.write(store_path)
        outputs["store"] = str(store_path)
        logger.info(f"OCED store saved: {store_path}")

    if cfg.materialize:
        logger.info(f"Materializing with {cfg.reasoner_backend}..")
        with _timed(timings, "reason"):
            Reasoner(
                str(onto_path),
                backend=cfg.reasoner_backend,
                profile=settings.profile,
            ).materialize()
        logger.info("Materialization complete.")

    stats = {**_store_stats(store), **_ontology_stats(semlog)}
    manifest = RunManifest(
        input_path=input_path,
        settings=settings,
        case_concept_source=source,
        outputs=outputs,
        stats=stats,
        timings=timings,
        config_file=cfg.config_file,
        command=cfg.command,
        materialized=cfg.materialize,
    )
    record = manifest.write(cfg.manifest_path(input_path))
    logger.info(f"Manifest saved: {cfg.manifest_path(input_path)}")
    return record


def run_build(cfg: RunConfig) -> BuildReport:
    """Build every input in ``cfg``, continuing past a file that fails."""
    for message in cfg.warnings:
        logger.warning(message)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    report = BuildReport()
    limit = cfg.size_limit_mb * _MB

    with FileLog(cfg.log_dir) as file_log:
        for input_path in cfg.inputs:
            size = input_path.stat().st_size
            if limit and size > limit:
                reason = f"exceeds the {cfg.size_limit_mb} MB limit ({size // _MB} MB)"
                logger.info(f"Skipping {input_path.name}: {reason}")
                report.skipped.append((input_path, reason))
                continue

            log_path = file_log.switch_to(cfg.output_stem(input_path))
            logger.info(f"=== Processing {input_path.name} ===")
            if log_path:
                logger.info(f"Log:    {log_path}")

            try:
                report.manifests.append(build_one(cfg, input_path))
                report.built.append(input_path)
                logger.info(f"=== Finished {input_path.name} ===")
            except Exception as exc:
                report.failed.append((input_path, str(exc)))
                logger.exception(f"Failed while processing {input_path.name}: {exc}")

    return report


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@dataclass
class InspectReport:
    """A read-only summary of one OCEL file, as ``semlog inspect`` prints it."""

    input_path: Path
    activities: list[tuple[str, int]]
    object_types: list[tuple[str, int]]
    relations: list[tuple[str, int]]
    time_range: tuple[str, str] | None
    suggested_case_concepts: list[str]
    renamed: dict[str, str]

    def render(self) -> str:
        lines = [f"{self.input_path}", ""]
        lines.append(f"activities ({len(self.activities)})")
        lines += [f"  {n:>9,}  {name}" for name, n in self.activities]
        lines.append("")
        lines.append(f"object types ({len(self.object_types)})")
        lines += [f"  {n:>9,}  {name}" for name, n in self.object_types]
        lines.append("")
        lines.append(f"relations ({len(self.relations)})")
        lines += [f"  {n:>9,}  {name}" for name, n in self.relations]
        lines.append("")
        if self.time_range is not None:
            lines.append(
                f"time range           {self.time_range[0]} .. {self.time_range[1]}"
            )
        lines.append(
            "suggested            --case-concepts "
            + " ".join(self.suggested_case_concepts)
        )
        if self.renamed:
            lines.append("")
            lines.append("names changed by cleaning (use the left-hand form)")
            lines += [
                f"  {clean:<28} was {original}"
                for clean, original in sorted(self.renamed.items())
            ]
        return "\n".join(lines)


def inspect_input(
    path: Path,
    extensions: Sequence[OcelExtension] = (),
    suggest: int = 3,
) -> InspectReport:
    """Summarise an OCEL file without building anything.

    Names are reported in their cleaned form, because that is what
    ``--case-concepts`` matches against, with the original shown alongside
    whenever cleaning changed it.
    """
    store = load_and_clean(path, extensions)

    activities = sorted(
        ((a, len(store.get_events(a).data)) for a in store.get_activities()),
        key=lambda item: item[1],
        reverse=True,
    )
    object_types = sorted(
        ((t, len(table.data)) for t, table in store.get_objects().items()),
        key=lambda item: item[1],
        reverse=True,
    )
    relations = sorted(
        (
            (name, len(store.get_relation(name).data))
            for name in store.get_relation_types()
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    stamps = [
        store.get_events(a).data[store.get_events(a).time]
        for a in store.get_activities()
    ]
    non_empty = [s for s in stamps if len(s)]
    time_range = (
        (str(min(s.min() for s in non_empty)), str(max(s.max() for s in non_empty)))
        if non_empty
        else None
    )

    schema_names = [a for a, _ in activities] + [t for t, _ in object_types]
    renamed = {
        name: original
        for name in schema_names
        if (original := store.get_original(name)) is not None and original != name
    }

    return InspectReport(
        input_path=path,
        activities=activities,
        object_types=object_types,
        relations=relations,
        time_range=time_range,
        suggested_case_concepts=store.get_most_common_objects(n=suggest),
        renamed=renamed,
    )


# ---------------------------------------------------------------------------
# mine-declare
# ---------------------------------------------------------------------------


def mine_declare(
    input_path: Path,
    output_path: Path,
    case_concepts: Sequence[str] | None,
    constraints: Sequence[str] | None,
    constraint_support: float,
    itemset_support: float,
    auto_case_concepts: int = 3,
    extensions: Sequence[OcelExtension] = (),
) -> Path:
    """Mine DECLARE constraints from one OCEL file into a CSV.

    Kept out of ``build`` on purpose: mining needs the Lydia backend, is far
    slower than a build, and its output is worth keeping as an artifact in its
    own right rather than recomputing per ontology.

    Returns:
        The path the constraints CSV was written to.
    """
    from semanticlog.enrich.declare import DeclareEnricher

    logger.info(f"Loading OCEL data from {input_path}..")
    store = load_and_clean(input_path, extensions)

    chosen = (
        list(case_concepts)
        if case_concepts is not None
        else store.get_most_common_objects(n=auto_case_concepts)
    )
    logger.info(f"Case concepts: {chosen}")

    logger.info("Mining DECLARE constraints via discovery..")
    result = DeclareEnricher(store).get_constraints_via_discovery(
        case_concepts=chosen,
        constraint_set=list(constraints) if constraints else None,
        constraint_support_threshold=constraint_support,
        itemset_support_threshold=itemset_support,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write(str(output_path))
    logger.info(f"{len(result.constraints)} constraints saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# reason
# ---------------------------------------------------------------------------


def reason(
    onto_path: Path,
    profile: OwlProfile,
    backend: str = "HermiT",
    output_path: Path | None = None,
    check_only: bool = False,
) -> bool:
    """Materialise or consistency-check an existing ontology file.

    Returns:
        ``True`` unless ``check_only`` found the ontology inconsistent.
    """
    reasoner = Reasoner(str(onto_path), backend=backend, profile=profile)
    if check_only:
        consistent = reasoner.check()
        logger.info(f"{onto_path}: {'consistent' if consistent else 'INCONSISTENT'}")
        return consistent

    logger.info(f"Materializing {onto_path} with {backend}..")
    reasoner.materialize(overwrite_old=output_path is None)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reasoner.save_onto(str(output_path))
    logger.info(f"Materialization complete: {output_path or onto_path}")
    return True
