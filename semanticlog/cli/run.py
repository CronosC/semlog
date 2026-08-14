"""What each subcommand does.

These functions take already-resolved values, so a command can be run and
tested without a command line. `build` loads, cleans, chooses case concepts,
builds, writes, and records what it did.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semanticlog.cli.config import RunConfig
from semanticlog.cli.logs import FileLog
from semanticlog.core.enums import (
    OCELExtension,
    OWLProfile,
)
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.binning import DEFAULT_BIN_LABELS
from semanticlog.data.cleaning import clean_oced_data, object_type_renames
from semanticlog.data.flatten import case_concept_candidates
from semanticlog.data.loading.pm4py_loader import PM4PyLoader
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind
from semanticlog.onto.audit import warn_on_foreign_declare
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.result import BuildResult
from semanticlog.onto.sinks import AxiomSink, InMemorySink, StreamingSink

logger = logging.getLogger("semlog.run")

_MB = 1024 * 1024
_HASH_CHUNK = 1 << 20

PARTIAL_SUFFIX = ".partial"


def write_backmap(data: OCEDStore, path: Path) -> None:
    """Write the map of cleaned names back to what the log called them.

    The map is a build artifact rather than part of the data, so writing it
    belongs beside the ontology it explains.
    """
    path.write_text(
        json.dumps(data.backmap, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks.

    This identifies the exact input behind a result, and is what lets a build
    notice DECLARE constraints mined from a different file.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _timed(timings: dict[str, float], stage: str):
    """Record how long one stage took, even if it fails."""
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = time.perf_counter() - started


@dataclass
class BuildReport:
    """What happened to each input of a build.

    Attributes:
        built: Inputs that produced an ontology.
        skipped: Inputs that were not attempted, with the reason.
        failed: Inputs that were attempted and failed, with the reason.
        records: The provenance record of each successful build.
    """

    built: list[Path] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every attempted build succeeded. A skip is not a failure."""
        return not self.failed

    def summary(self) -> str:
        """Return a one-line count of what happened."""
        parts = [f"{len(self.built)} built"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts)


def load_and_clean(
    path: Path,
    extensions: Sequence[OCELExtension],
    bin_labels: Sequence[str] = DEFAULT_BIN_LABELS,
) -> OCEDStore:
    """Read one OCEL file and make its names safe to emit.

    Args:
        path: OCEL file to read.
        extensions: OCEL extensions to load as well.
        bin_labels: Bin labels the build will use, which cleaning reserves
            along with the rest of the names a build derives for itself. The
            default is what a build reserves when the labels are left alone,
            so `inspect` reports the names a build would use. A run with its
            own labels passes them, and only then can the two differ.
    """
    data = PM4PyLoader(str(path), {}, ocel_extensions=list(extensions)).get_oced()
    return clean_oced_data(data, bin_labels=bin_labels)


def follow_renames(data: OCEDStore, case_concepts: Sequence[str]) -> list[str]:
    """Map given case concepts onto the names the store holds them under.

    A case concept is written as the source log spells it, and cleaning renames
    an object type whose name something else in the log already claims. A name
    that matches no object type and no rename is passed through, so the audit
    is what reports it.

    Args:
        data: Store the names have to resolve against.
        case_concepts: Object types as they were given.

    Returns:
        The same types, in the same order, under their current names.
    """
    object_types = set(data.get_object_types())
    renames = object_type_renames(data)
    chosen = []
    for case_concept in case_concepts:
        moved = renames.get(case_concept)
        if case_concept in object_types or moved is None:
            chosen.append(case_concept)
            continue
        logger.info(f"  case concept {case_concept} is called {moved} after cleaning")
        chosen.append(moved)
    return chosen


def rank_case_concepts(data: OCEDStore, n: int) -> list[str]:
    """Return the n largest object types a build could project onto.

    Types no event-to-object relation reaches are left out before the count is
    applied, so choosing three never spends one of them on a type that has no
    events to order.

    Args:
        data: Store to choose from.
        n: How many types to return.

    Returns:
        Type names, most instances first.
    """
    reachable = set(case_concept_candidates(data))
    ranked = data.get_most_common_objects(n=len(data.get_object_types()))
    return [obj_type for obj_type in ranked if obj_type in reachable][:n]


def choose_case_concepts(cfg: RunConfig, data: OCEDStore) -> tuple[list[str], str]:
    """Take the configured case concepts, or pick the most common object types.

    Returns:
        The chosen types and where the choice came from, which is recorded as
        provenance so a result says whether it was chosen or given.
    """
    if cfg.case_concepts is not None:
        return follow_renames(data, cfg.case_concepts), "config"
    chosen = rank_case_concepts(data, cfg.auto_case_concepts)
    if not chosen and data.get_object_types():
        logger.warning(
            "No object type of this log is reached by an event-to-object relation, "
            "so there is nothing to project onto and no case concept was chosen."
        )
    return chosen, f"auto(n={cfg.auto_case_concepts})"


def _relation_rows(data: OCEDStore, kind: RelationKind) -> int:
    """Count the rows across one relation family."""
    return sum(
        len(data.get_relation(name).data) for name in data.get_relation_types(kind)
    )


def _store_stats(data: OCEDStore) -> dict[str, int]:
    """Measure the loaded log, for the provenance record."""
    rows = _rows_by_kind(data)
    events, objects = rows[EntityKind.EVENT], rows[EntityKind.OBJECT]
    return {
        "activities": len(events),
        "events": sum(n for _, n in events),
        "object_types": len(objects),
        "objects": sum(n for _, n in objects),
        "event_to_object_relations": _relation_rows(data, RelationKind.E2O),
        "object_to_object_relations": _relation_rows(data, RelationKind.O2O),
    }


def _rows_by_kind(data: OCEDStore) -> dict[EntityKind | None, list[tuple[str, int]]]:
    """Return each entity type with its row count, grouped by which half it is in."""
    grouped: dict[EntityKind | None, list[tuple[str, int]]] = defaultdict(list)
    for view in data.tables(changes=False):
        grouped[view.kind].append((view.name, len(view.frame)))
    return grouped


def _ontology_stats(semlog: BuildResult, sink: AxiomSink) -> dict[str, int]:
    """Measure the built ontology, however it was built.

    The counts come from the sink's name ledger, so a streamed build and an
    in-memory one report the same keys and two runs can be compared. Only an
    in-memory build can count individuals, so that one is extra.
    """
    stats = {**sink.declared_by_kind(), "axioms": sink.axiom_count}
    if semlog.onto is not None:
        stats["individuals"] = sum(1 for _ in semlog.onto.individuals_in_signature())
    return stats


def run_record(
    cfg: RunConfig,
    input_path: Path,
    digest: str,
    case_concept_source: str,
    data: OCEDStore,
) -> dict[str, Any]:
    """Assemble what to record about this run.

    The result is annotated onto the ontology, so it has to say everything
    needed to identify the input and reproduce the run.

    Args:
        cfg: The resolved run.
        input_path: Log being built.
        digest: Checksum of that log.
        case_concept_source: Whether the case concepts were given or chosen.
        data: The loaded log, measured for the record.

    Returns:
        The record, keyed by annotation name. `materializeRequested` says what
        the run asked for, since the record is written into the ontology
        before the reasoner has had a chance to run.
    """
    outputs = {}
    if cfg.save_backmap:
        outputs["backmap"] = str(cfg.backmap_path(input_path))

    return {
        "sourceLogSha256": digest,
        "sourceLogBytes": input_path.stat().st_size,
        "caseConceptSource": case_concept_source,
        "buildCommand": cfg.command,
        "buildConfigFile": None if cfg.config_file is None else str(cfg.config_file),
        "buildInputStatistics": _store_stats(data),
        "buildOutputs": outputs or None,
        "materializeRequested": cfg.materialize,
    }


def build_one(cfg: RunConfig, input_path: Path) -> dict[str, Any]:
    """Build one input, from loading the log to writing the ontology.

    The provenance record is assembled before the build, so the ontology
    carries it from the start. The ontology itself goes to a `.partial` file
    that is renamed into place once it is complete, so a build that fails
    leaves whatever was there before it untouched.

    Args:
        cfg: The resolved run.
        input_path: Log to build.

    Returns:
        The provenance record of this build.
    """
    onto_path = cfg.ontology_path(input_path)
    timings: dict[str, float] = {}

    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {onto_path}")

    digest = file_digest(input_path)

    logger.info("Loading OCEL data..")
    with _timed(timings, "load"):
        data = load_and_clean(
            input_path, cfg.settings.ocel_extensions, cfg.settings.bin_labels
        )

    case_concepts, source = choose_case_concepts(cfg, data)
    logger.info(f"Case concepts ({source}): {case_concepts or 'none'}")
    settings = cfg.settings_for(case_concepts)
    warn_on_foreign_declare(settings, digest)

    logger.info("Building ontology..")
    onto_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = onto_path.with_name(onto_path.name + PARTIAL_SUFFIX)
    sink: AxiomSink
    if cfg.streaming:
        sink = StreamingSink(str(partial_path), DEFAULT_IRI, settings)
    else:
        sink = InMemorySink(DEFAULT_IRI, settings)

    record = run_record(cfg, input_path, digest, source, data)

    try:
        with _timed(timings, "build"):
            semlog = OCEDOntoBuilder(DEFAULT_IRI, settings, sink=sink).build_ontology(
                data, source=str(input_path), run=record
            )

        with _timed(timings, "write"):
            if semlog.onto is not None:
                semlog.save(str(partial_path))
            partial_path.replace(onto_path)
            if semlog.path is not None:
                semlog = dataclasses.replace(semlog, path=str(onto_path))
    except BaseException:
        sink.abort()
        partial_path.unlink(missing_ok=True)
        raise
    logger.info(f"Ontology saved: {onto_path}")

    if cfg.save_backmap:
        backmap_path = cfg.backmap_path(input_path)
        write_backmap(data, backmap_path)
        logger.info(f"Name backmap saved: {backmap_path}")

    if cfg.materialize:
        logger.info(f"Materializing with {cfg.reasoner_backend}..")
        with _timed(timings, "reason"):
            Reasoner(
                str(onto_path),
                backend=cfg.reasoner_backend,
                profile=settings.profile,
            ).materialize()
        logger.info("Materialization complete.")

    stats = _ontology_stats(semlog, sink)
    logger.info("Ontology: " + ", ".join(f"{k}={v:,}" for k, v in stats.items()))
    logger.info("Timings:  " + ", ".join(f"{k}={v:.2f}s" for k, v in timings.items()))
    return record


def build_all(cfg: RunConfig) -> BuildReport:
    """Build every input of a run.

    One input failing does not stop the others, since a run over a directory
    is usually worth finishing. Every failure is recorded and reported at the
    end.

    Args:
        cfg: The resolved run.

    Returns:
        What happened to each input.
    """
    for message in cfg.warnings:
        logger.warning(message)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    report = BuildReport()
    limit = cfg.size_limit_mb * _MB

    with FileLog(cfg.log_dir) as file_log:
        for input_path in cfg.inputs:
            size = input_path.stat().st_size
            if limit and size > limit:
                skipped = f"exceeds the {cfg.size_limit_mb} MB limit ({size // _MB} MB)"
                logger.info(f"Skipping {input_path.name}: {skipped}")
                report.skipped.append((input_path, skipped))
                continue

            log_path = file_log.switch_to(cfg.output_stem(input_path))
            logger.info(f"=== Processing {input_path.name} ===")
            if log_path:
                logger.info(f"Log:    {log_path}")

            try:
                report.records.append(build_one(cfg, input_path))
                report.built.append(input_path)
                logger.info(f"=== Finished {input_path.name} ===")
            except Exception as exc:
                report.failed.append((input_path, str(exc)))
                logger.exception(f"Failed while processing {input_path.name}: {exc}")

    return report


@dataclass
class InspectReport:
    """What one log contains, as `semlog inspect` reports it.

    Attributes:
        input_path: The log this describes.
        activities: Activity name and event count, most events first.
        object_types: Object type and object count, most objects first.
        relations: Relation name and row count, most rows first.
        time_range: Earliest and latest timestamp, or None if there are none.
        suggested_case_concepts: What a build would choose as case concepts.
        renamed: Names cleaning changed, as cleaned name to source name.
    """

    input_path: Path
    activities: list[tuple[str, int]]
    object_types: list[tuple[str, int]]
    relations: list[tuple[str, int]]
    time_range: tuple[str, str] | None
    suggested_case_concepts: list[str]
    renamed: dict[str, str]

    def render(self) -> str:
        """Return the report as the text the command prints."""
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
    extensions: Sequence[OCELExtension] = (),
    suggest: int = 3,
) -> InspectReport:
    """Summarize one log without building anything.

    The log is cleaned first, so the names shown are the ones a build would
    use and the ones `--case-concepts` expects.

    Args:
        path: Log to summarize.
        extensions: OCEL extensions to load as well.
        suggest: How many case concepts to suggest.

    Returns:
        The summary.
    """
    data = load_and_clean(path, extensions)

    def by_count(counted: list[tuple[str, int]]) -> list[tuple[str, int]]:
        return sorted(counted, key=lambda item: item[1], reverse=True)

    rows = _rows_by_kind(data)
    activities = by_count(rows[EntityKind.EVENT])
    object_types = by_count(rows[EntityKind.OBJECT])
    relations = by_count(
        [
            (name, len(data.get_relation(name).data))
            for name in data.get_relation_types()
        ]
    )

    stamps = [
        view.frame[column]
        for view in data.tables(changes=False)
        if view.kind is EntityKind.EVENT
        for column in view.structural
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
        if (original := data.get_original(name)) is not None and original != name
    }

    return InspectReport(
        input_path=path,
        activities=activities,
        object_types=object_types,
        relations=relations,
        time_range=time_range,
        suggested_case_concepts=rank_case_concepts(data, suggest),
        renamed=renamed,
    )


def mine_declare(
    input_path: Path,
    output_path: Path,
    case_concepts: Sequence[str] | None,
    constraints: Sequence[str] | None,
    constraint_support: float,
    itemset_support: float,
    auto_case_concepts: int = 3,
    extensions: Sequence[OCELExtension] = (),
) -> Path:
    """Mine DECLARE constraints from a log and write them to a CSV.

    The origin of the constraints is recorded with them, so a later build can
    tell whether they belong to the log it is building.

    Args:
        input_path: Log to mine.
        output_path: CSV to write.
        case_concepts: Object types to project onto, or None to choose them.
        constraints: Constraint templates to keep, or None to keep everything
            discovery proposes.
        constraint_support: Share of traces a constraint has to hold in.
        itemset_support: Share of traces an activity set has to occur in.
        auto_case_concepts: How many case concepts to choose when choosing.
        extensions: OCEL extensions to load as well.

    Returns:
        The file written.

    Raises:
        ImportError: If declare4py or its solver is not installed.
    """
    from semanticlog.enrich.declare import DeclareEnricher

    logger.info(f"Loading OCEL data from {input_path}..")
    data = load_and_clean(input_path, extensions)

    chosen = (
        follow_renames(data, case_concepts)
        if case_concepts is not None
        else rank_case_concepts(data, auto_case_concepts)
    )
    logger.info(f"Case concepts: {chosen}")

    logger.info("Mining DECLARE constraints via discovery..")
    result = DeclareEnricher(data).get_constraints_via_discovery(
        case_concepts=chosen,
        constraint_set=list(constraints) if constraints else None,
        constraint_support_threshold=constraint_support,
        itemset_support_threshold=itemset_support,
    )
    result = dataclasses.replace(
        result, source_log=str(input_path), source_sha256=file_digest(input_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write(str(output_path))
    logger.info(f"{len(result.constraints)} constraints saved: {output_path}")
    return output_path


def reason(
    onto_path: Path,
    profile: OWLProfile,
    backend: str = "HermiT",
    output_path: Path | None = None,
    check_only: bool = False,
) -> bool:
    """Run a reasoner over an ontology that already exists.

    Args:
        onto_path: Ontology to read.
        profile: Profile it was built for, which decides what is asked of the
            reasoner.
        backend: Reasoner to use.
        output_path: Where to write the materialized result. Without one, the
            input file is overwritten.
        check_only: Only report consistency, changing nothing.

    Returns:
        Whether the ontology is consistent under `check_only`, and True
        otherwise.
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
        reasoner.save(str(output_path))
    logger.info(f"Materialization complete: {output_path or onto_path}")
    return True
