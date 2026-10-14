"""The bodies of the `build`, `mine-declare` and `reason` commands.

These functions take resolved values. Start at `build_all`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semanticlog.cli.config import RunConfig
from semanticlog.cli.logs import FileLog
from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import OCELExtension
from semanticlog.core.errors import SemlogError
from semanticlog.data.flatten import choose_case_concepts
from semanticlog.data.loading import load_and_clean
from semanticlog.data.readable import ReadableStore
from semanticlog.data.stats import file_digest, store_stats
from semanticlog.enrich.declare import DeclareResult
from semanticlog.onto.emitters.provenance import build_configuration
from semanticlog.onto.pipeline import build_log
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.record import recorded_annotation, recorded_configuration
from semanticlog.onto.result import BuildResult
from semanticlog.onto.settings import OntoBuildSettings

logger = logging.getLogger(__name__)

_MB = 1024 * 1024


def _save_backmap(data: ReadableStore, path: Path) -> None:
    """Write the map of cleaned names to source names as JSON."""
    path.write_text(
        json.dumps(data.backmap, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def warn_on_foreign_declare(declare: DeclareResult | None, digest: str) -> None:
    """Warn when the DECLARE constraints were mined from a different file.

    The audit checks whether the trace ids resolve.
    """
    if declare is None or declare.source_sha256 is None:
        return
    if declare.source_sha256 != digest:
        logger.warning(
            "DECLARE constraints were mined from %r "
            "(sha256 %s), which is not the file being "
            "built (sha256 %s).",
            declare.source_log,
            declare.source_sha256[:12],
            digest[:12],
        )


def _warn_on_changed_settings(
    onto_path: Path, settings: OntoBuildSettings, case_concepts: Sequence[str]
) -> None:
    """Warn when the file about to be written holds a build of other settings.

    The case concepts are compared along with the settings.
    """
    recorded = recorded_configuration(onto_path)
    if recorded is None:
        return
    recorded = {
        **recorded,
        "case_concepts": recorded_annotation(onto_path, vocab.CASE_CONCEPTS),
    }
    current = {**build_configuration(settings), "case_concepts": list(case_concepts)}
    changed = sorted(
        key
        for key in set(recorded) | set(current)
        if recorded.get(key) != current.get(key)
    )
    if changed:
        logger.warning(
            "%s holds a build with different settings (%s) "
            "and is about to be overwritten. Pass --name to keep both.",
            onto_path,
            ", ".join(changed),
        )


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


def _ontology_stats(result: BuildResult) -> dict[str, int]:
    """Return the counts of the built ontology, with individuals if held in memory."""
    stats = dict(result.stats)
    if result.onto is not None:
        stats["individuals"] = sum(1 for _ in result.onto.individuals_in_signature())
    return stats


def run_record(
    cfg: RunConfig,
    input_path: Path,
    digest: str,
    case_concepts: Sequence[str],
    case_concept_source: str,
    data: ReadableStore,
) -> dict[str, Any]:
    """Return what this run resolved per log, keyed by annotation name.

    The settings themselves go under `buildConfiguration`. The record is
    written before the reasoner runs, so `materializeRequested` holds what the
    run asked for.

    Args:
        digest: SHA-256 of the input.
        case_concept_source: Whether the case concepts were given or chosen.
    """
    outputs = {}
    if cfg.save_backmap:
        outputs["backmap"] = str(cfg.backmap_path(input_path))

    return {
        vocab.SOURCE_LOG_SHA256: digest,
        vocab.SOURCE_LOG_BYTES: input_path.stat().st_size,
        vocab.CASE_CONCEPTS: list(case_concepts) or None,
        vocab.CASE_CONCEPT_SOURCE: case_concept_source,
        vocab.DECLARE_FILE: cfg.declare_file,
        vocab.ANNOTATION_DOCUMENT: cfg.annotation_document,
        vocab.BUILD_COMMAND: cfg.command,
        vocab.BUILD_CONFIG_FILE: None
        if cfg.config_file is None
        else str(cfg.config_file),
        vocab.BUILD_INPUT_STATISTICS: store_stats(data),
        vocab.BUILD_OUTPUTS: outputs or None,
        vocab.MATERIALIZE_REQUESTED: cfg.materialize,
    }


def build_one(cfg: RunConfig, input_path: Path) -> dict[str, Any]:
    """Build one input, from loading the log to writing the ontology.

    Returns:
        The provenance record of this build.
    """
    onto_path = cfg.ontology_path(input_path)
    timings: dict[str, float] = {}

    logger.info("Input:  %s", input_path)
    logger.info("Output: %s", onto_path)

    digest = file_digest(input_path)

    logger.info("Loading event data..")
    with _timed(timings, "load"):
        data = load_and_clean(
            input_path,
            cfg.settings.ocel_extensions,
            cfg.settings.bin_labels,
            cfg.loader_columns(),
        )

    case_concepts, source = choose_case_concepts(
        data, cfg.case_concepts, cfg.auto_case_concepts
    )
    logger.info("Case concepts (%s): %s", source, case_concepts or "none")
    declare = cfg.declare_result()
    warn_on_foreign_declare(declare, digest)
    inputs = cfg.build_inputs(case_concepts, declare)

    settings = cfg.settings
    _warn_on_changed_settings(onto_path, settings, case_concepts)
    record = run_record(cfg, input_path, digest, case_concepts, source, data)

    logger.info("Building ontology..")
    with _timed(timings, "build"):
        result = build_log(
            data,
            settings,
            inputs,
            onto_path,
            iri=cfg.iri,
            source=str(input_path),
            record=record,
            streaming=cfg.streaming,
        )
    logger.info("Ontology saved: %s", onto_path)

    if cfg.save_backmap:
        backmap_path = cfg.backmap_path(input_path)
        _save_backmap(data, backmap_path)
        logger.info("Name backmap saved: %s", backmap_path)

    if cfg.materialize:
        logger.info("Materializing with %s..", cfg.reasoner_backend)
        with _timed(timings, "reason"):
            added = Reasoner(str(onto_path), backend=cfg.reasoner_backend).materialize()
        logger.info("Materialization complete: %s axiom(s) added.", f"{added:,}")

    stats = _ontology_stats(result)
    logger.info("Ontology: %s", ", ".join(f"{k}={v:,}" for k, v in stats.items()))
    logger.info("Timings:  %s", ", ".join(f"{k}={v:.2f}s" for k, v in timings.items()))
    return record


def build_all(cfg: RunConfig) -> BuildReport:
    """Build every input of a run and return what happened to each.

    A failed input is recorded and the next one is built. A `SemlogError` or
    `OSError` is logged as one line. Any other exception is logged with its
    traceback.
    """
    for message in cfg.warnings:
        logger.warning(message)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    report = BuildReport()
    limit = cfg.size_limit_mb * _MB

    with FileLog(cfg.log_dir) as file_log:
        for input_path in cfg.inputs:
            try:
                size = input_path.stat().st_size
                if limit and size > limit:
                    skipped = (
                        f"exceeds the {cfg.size_limit_mb} MB limit ({size // _MB} MB)"
                    )
                    logger.info("Skipping %s: %s", input_path.name, skipped)
                    report.skipped.append((input_path, skipped))
                    continue

                log_path = file_log.switch_to(cfg.output_stem(input_path))
                logger.info("=== Processing %s ===", input_path.name)
                if log_path:
                    logger.info("Log:    %s", log_path)

                report.records.append(build_one(cfg, input_path))
                report.built.append(input_path)
                logger.info("=== Finished %s ===", input_path.name)
            except (SemlogError, OSError) as exc:
                report.failed.append((input_path, str(exc)))
                logger.error("Failed while processing %s: %s", input_path.name, exc)
            except Exception as exc:
                report.failed.append((input_path, str(exc)))
                logger.exception("Failed while processing %s: %s", input_path.name, exc)

    return report


def mine_declare(
    input_path: Path,
    output_path: Path | None = None,
    case_concepts: Sequence[str] | None = None,
    constraints: Sequence[str] | None = None,
    constraint_support: float = 0.5,
    itemset_support: float = 0.5,
    auto_case_concepts: int = 3,
    extensions: Sequence[OCELExtension] = (),
    columns: Mapping[str, Any] | None = None,
) -> Path:
    """Mine DECLARE constraints from a log and write them to a CSV.

    The source log and its checksum are recorded with the constraints. The
    templates are checked before the log is read.

    Args:
        output_path: CSV to write, or None for `<input stem>.csv` in the
            current directory.
        case_concepts: Object types to project onto, or None to choose them.
        constraints: Constraint templates to keep, or None to keep everything
            discovery proposes.
        constraint_support: Share of traces a constraint has to hold in.
        itemset_support: Share of traces an activity set has to occur in.
        auto_case_concepts: How many case concepts to choose when choosing.
        columns: Column names for a flat `.csv` or `.xes` log.

    Returns:
        The file written.

    Raises:
        DependencyError: If Declare4Py or its solver is not installed.
        ConfigError: If a constraint template is not one Declare4Py knows.
    """
    from semanticlog.enrich.declare import DeclareEnricher, check_templates

    check_templates(list(constraints or []))
    if output_path is None:
        output_path = Path(f"{input_path.stem}.csv")

    logger.info("Loading event data from %s..", input_path)
    data = load_and_clean(input_path, extensions, columns=columns)

    chosen, _ = choose_case_concepts(data, case_concepts, auto_case_concepts)
    logger.info("Case concepts: %s", chosen)

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
    logger.info("%s constraints saved: %s", len(result.constraints), output_path)
    return output_path


def reason(
    onto_path: Path,
    backend: str = "HermiT",
    output_path: Path | None = None,
    check_only: bool = False,
) -> bool:
    """Run a reasoner over a built ontology.

    Args:
        output_path: Where to write the materialized result. Without one, the
            input file is overwritten.
        check_only: Report consistency and write nothing.

    Returns:
        Whether the ontology is consistent under `check_only`, otherwise True.
    """
    reasoner = Reasoner(str(onto_path), backend=backend)
    if check_only:
        return reasoner.check()

    logger.info("Materializing %s with %s..", onto_path, backend)
    added = reasoner.materialize(overwrite_old=output_path is None)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reasoner.save(str(output_path))
    logger.info(
        "Materialization complete: %s axiom(s) added to %s",
        f"{added:,}",
        output_path or onto_path,
    )
    return True
