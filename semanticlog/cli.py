import argparse
import glob
import logging
import logging.config
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.loading.pm4pyloader import PM4PYLoader
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OcelExtension,
    OntoBuildSettings,
    OwlProfile,
)

LOG_KEEP = 10
_LOG_FORMAT = "%(name)s | %(asctime)s | %(message)s"
_LOG_DATEFMT = "%H:%M:%S"

logger = logging.getLogger("oced.pipeline")


def _setup_console_logging() -> None:
    logging.config.dictConfig({"version": 1, "disable_existing_loggers": True})
    logging.root.handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)


class FileLog:
    """Owns the per-input file log handler, so that a run holds no module state.

    Args:
        log_dir: Directory to write logs into. Empty string disables file logging.
        keep: How many previous logs to retain per input stem.
    """

    def __init__(self, log_dir: str, keep: int = LOG_KEEP) -> None:
        self.log_dir = log_dir
        self.keep = keep
        self._handler: logging.FileHandler | None = None

    def switch_to(self, stem: str) -> str | None:
        """Route logging to a fresh file for ``stem`` and prune old ones.

        Returns:
            The new log file path, or ``None`` when file logging is disabled.
        """
        if not self.log_dir:
            return None
        os.makedirs(self.log_dir, exist_ok=True)
        self.close()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.log_dir, f"{ts}_{stem}.log")
        self._handler = logging.FileHandler(log_path)
        self._handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
        logging.root.addHandler(self._handler)

        existing = sorted(glob.glob(os.path.join(self.log_dir, f"*_{stem}.log")))
        for old in existing[: -self.keep]:
            os.remove(old)

        return log_path

    def close(self) -> None:
        """Detach and close the current handler, if any."""
        if self._handler is not None:
            logging.root.removeHandler(self._handler)
            self._handler.close()
            self._handler = None


class _OutputPaths(NamedTuple):
    onto: str
    id_map: str
    oced: str
    declare: str


def _output_paths(stem: str, args: argparse.Namespace) -> _OutputPaths:
    mat_str = "_MAT" if args.materialize else ""
    order_str = "_".join(args.order) if args.order else "NoOrder"
    ext_str = "_" + "_".join(args.ocel_extensions) if args.ocel_extensions else ""
    data_repr_str = (
        "_" + "_".join(sorted(r.upper() for r in args.data_representation))
        if args.data_representation
        else "_NONE"
    )
    annotated_str = (
        "_ANNOTATED" if (args.annotation_document or args.derive_annotations) else ""
    )
    base = f"{stem}{mat_str}_{args.profile}_{order_str}{data_repr_str}{ext_str}{annotated_str}"
    return _OutputPaths(
        onto=os.path.join(args.output, f"{base}.owl"),
        id_map=os.path.join(args.id_map_output, f"{base}.json"),
        oced=os.path.join(args.oced_output, f"{base}.pkl"),
        declare=os.path.join(args.output, f"{base}_declare.csv"),
    )


def _build_settings(
    args: argparse.Namespace,
    case_concepts: list[str],
    ocel_extensions: list[OcelExtension],
) -> OntoBuildSettings:
    settings = OntoBuildSettings()
    settings.profile = OwlProfile[args.profile]
    settings.order_representation = [EventOrderRepresentation[o] for o in args.order]
    settings.ocel_extensions = ocel_extensions
    settings.inverse_representation = [
        InverseRepresentation[r] for r in args.inverse_representation
    ]
    settings.case_concepts = case_concepts
    settings.data_representation = [
        DataRepresentation[r.upper()] for r in args.data_representation
    ]
    settings.bins = args.bins
    if args.bin_labels is not None:
        settings.bin_labels = args.bin_labels
    settings.annotation_document = args.annotation_document
    settings.derive_annotations = args.derive_annotations
    return settings


def _process_file(file_path: str, stem: str, args: argparse.Namespace) -> None:
    paths = _output_paths(stem, args)
    ocel_extensions = [OcelExtension[e] for e in args.ocel_extensions]

    logger.info(f"Input:  {file_path}")
    logger.info(f"Output: {paths.onto}")

    logger.info("Loading OCEL data..")
    oced = PM4PYLoader(file_path, {}, ocel_extensions=ocel_extensions).get_oced()
    logger.info("Cleaning OCED data..")
    oced = clean_oced_data(oced)
    if args.case_concepts is not None:
        case_concepts = args.case_concepts
    else:
        logger.info("Selecting most common objects as case concepts (n=3)..")
        case_concepts = oced.get_most_common_objects(n=3)
    logger.info(f"Case concepts: {case_concepts}")

    settings = _build_settings(args, case_concepts, ocel_extensions)

    if settings.instantiate_declare and args.declare_constraints is not None:
        from semanticlog.enrich.declare import DeclareEnricher

        logger.info("Mining DECLARE constraints via discovery..")
        enricher = DeclareEnricher(oced)
        support = (
            args.declare_constraint_support
            if args.declare_constraint_support is not None
            else 0.5
        )
        itemset_support = (
            args.declare_itemset_support
            if args.declare_itemset_support is not None
            else 0.5
        )
        declare_result = enricher.get_constraints_via_discovery(
            case_concepts=case_concepts,
            constraint_set=args.declare_constraints,
            constraint_support_threshold=support,
            itemset_support_threshold=itemset_support,
        )
        declare_result.write(paths.declare)
        logger.info(f"DECLARE constraints saved: {paths.declare}")
        settings.declare_file = paths.declare
        settings.declare_result = declare_result

    logger.info("Building ontology..")
    semlog = OCEDOntoBuilder(DEFAULT_IRI, settings).build_ontology(oced)
    semlog.write_ontology(paths.onto)
    logger.info(f"Ontology saved: {paths.onto}")

    oced.save_backmap(paths.id_map)
    logger.info(f"ID backmap saved: {paths.id_map}")
    oced.save(paths.oced)
    logger.info(f"OCED store saved: {paths.oced}")

    if args.materialize:
        logger.info("Materializing with reasoner..")
        Reasoner(paths.onto, profile=OwlProfile[args.profile]).materialize()
        logger.info("Materialization complete.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OWL ontologies from OCEL event log files (OCEL 2.0: .xml, .json, .sqlite).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              # process all files in a folder with defaults
              semlog-build --folder data/ocel/

              # specific file, OWL2EL profile, distribution-fitted binning
              semlog-build --folder data/ocel/ --files Hiring.xml \\
                  --profile OWL2EL --data-representation dist_bins

              # include DECLARE constraints (requires Lydia backend)
              semlog-build --folder data/ocel/ --files Hiring.xml \\
                  --order DF DECLARE --declare-constraints Existence1 Response
            """),
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to folder containing OCEL 2.0 input files (.xml, .xmlocel, .json, .jsonocel, .sqlite).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Filenames within --folder to process. Omit to process all files.",
    )
    parser.add_argument(
        "--output",
        default="data/onto",
        help="Output directory for built ontologies (default: data/onto).",
    )
    parser.add_argument(
        "--id-map-output",
        default="data/id_map",
        help="Output directory for id_backmap JSON files (default: data/id_map).",
    )
    parser.add_argument(
        "--oced-output",
        default="data/oced",
        help="Output directory for serialized OCEDStore files (default: data/oced).",
    )
    parser.add_argument(
        "--profile",
        choices=["OWL2DL", "OWL2EL"],
        default="OWL2DL",
        help="OWL profile to use (default: OWL2DL).",
    )
    parser.add_argument(
        "--order",
        nargs="*",
        choices=["DF", "EF", "EF_explicit", "DECLARE"],
        default=["DF"],
        help="Event order representations to include (default: DF).",
    )
    parser.add_argument(
        "--materialize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run reasoner to materialize inferred triples (default: on).",
    )
    parser.add_argument(
        "--size-limit",
        type=int,
        default=0,
        metavar="MB",
        help="Skip files larger than this size in MB (default: 0 = no limit).",
    )
    parser.add_argument(
        "--ocel-extensions",
        nargs="*",
        choices=["ATTRIBUTE_HISTORY"],
        default=[],
        metavar="EXT",
        help=(
            "OCEL 2.0 extensions to enable. Choices: ATTRIBUTE_HISTORY "
            "(object attribute change history). Default: none."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default="",
        help="Directory for file logs (default: '' = no file logging).",
    )
    parser.add_argument(
        "--data-representation",
        nargs="*",
        choices=["values", "equal_bins", "dist_bins"],
        default=["values"],
        metavar="MODE",
        help=(
            "How numeric attributes are represented in the ontology. "
            "values: keep raw numeric values. "
            "equal_bins: bin into equal-frequency buckets. "
            "dist_bins: fit a distribution and bin by quantile (falls back to equal_bins). "
            "Pass no values to omit numeric attributes entirely. "
            "equal_bins and dist_bins are mutually exclusive. "
            "Default: values."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=5,
        metavar="N",
        help="Number of bins for equal_bins/dist_bins modes (default: 5).",
    )
    parser.add_argument(
        "--bin-labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help=(
            "Bin labels for binning modes. Must have exactly --bins elements. "
            "Defaults to very_low low average high very_high when --bins is 5."
        ),
    )
    parser.add_argument(
        "--inverse-representation",
        nargs="*",
        default=[],
        choices=["AXIOMATIC", "IMPLICIT"],
        metavar="STRATEGY",
        help=(
            "Inverse relation strategies to include: AXIOMATIC (InverseObjectProperty "
            "TBox axiom, OWL2DL only) and/or IMPLICIT (ABox materialisation). "
            "Pass no values (--inverse-representation) to omit inverses entirely. "
            "Default: none (no inverse relations)."
        ),
    )
    parser.add_argument(
        "--case-concepts",
        nargs="*",
        default=None,
        metavar="CONCEPT",
        help="Object types to use as case concepts (traces). Omit to auto-select the 3 most frequent object types.",
    )
    parser.add_argument(
        "--annotation-document",
        default=None,
        metavar="PATH",
        help="Path to a JSON file mapping entity names to {label, comment} dicts for ABox class annotations.",
    )
    parser.add_argument(
        "--derive-annotations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit rdfs:label on event, Time, object, and attribute individuals derived from source data (default: off).",
    )
    parser.add_argument(
        "--declare-constraints",
        nargs="+",
        default=None,
        metavar="CONSTRAINT",
        help=(
            "DECLARE constraint types to mine (e.g. Existence1 Response Precedence). "
            "Requires DECLARE in --order and the Lydia backend "
            "(install with: uv sync --extra declare). "
            "Triggers constraint mining via DeclareEnricher.get_constraints_via_discovery."
        ),
    )
    parser.add_argument(
        "--declare-constraint-support",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help=(
            "Minimum per-constraint support threshold for DECLARE discovery (default: 0.5). "
            "Requires --declare-constraints."
        ),
    )
    parser.add_argument(
        "--declare-itemset-support",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help=(
            "Minimum itemset support threshold for DECLARE discovery (default: 0.5). "
            "Requires --declare-constraints."
        ),
    )
    return parser.parse_args(argv)


def select_input_files(args: argparse.Namespace) -> list[str]:
    """Return the filenames in ``args.folder`` this run should process."""
    all_files = [
        f
        for f in os.listdir(args.folder)
        if os.path.isfile(os.path.join(args.folder, f))
    ]
    return [f for f in all_files if f in args.files] if args.files else all_files


def validate_args(args: argparse.Namespace) -> None:
    """Check the flag combination before any input file is read.

    Configuration conflicts are delegated to ``OntoBuildSettings.validate()`` so the
    rules live in one place; only argparse-level constraints are checked here.

    Raises:
        SystemExit: If the arguments are inconsistent.
    """
    declare_args_given = (
        args.declare_constraints is not None
        or args.declare_constraint_support is not None
        or args.declare_itemset_support is not None
    )
    if declare_args_given and "DECLARE" not in args.order:
        raise SystemExit(
            "error: --declare-constraints/--declare-constraint-support/"
            "--declare-itemset-support require DECLARE in --order."
        )

    try:
        probe = _build_settings(
            args, [], [OcelExtension[e] for e in args.ocel_extensions]
        )
        probe.validate(require_inputs=False)
    except ValueError as e:
        raise SystemExit(f"error: {e}") from e

    dr = set(args.data_representation)
    has_bins = bool(dr & {"equal_bins", "dist_bins"})
    if not has_bins and (args.bins != 5 or args.bin_labels is not None):
        logger.warning(
            "--bins/--bin-labels have no effect when "
            f"--data-representation={args.data_representation}"
        )


def main(argv: list[str] | None = None) -> None:
    _setup_console_logging()
    args = parse_args(argv)
    validate_args(args)

    size_limit_bytes = args.size_limit * 1024 * 1024 if args.size_limit > 0 else 0
    target_files = select_input_files(args)

    for output_dir in (args.output, args.id_map_output, args.oced_output):
        os.makedirs(output_dir, exist_ok=True)

    file_log = FileLog(args.log_dir)
    failed: list[str] = []
    for file_name in target_files:
        file_path = os.path.join(args.folder, file_name)

        if size_limit_bytes and os.path.getsize(file_path) > size_limit_bytes:
            logger.info(f"Skipping {file_name} (exceeds {args.size_limit} MB limit)")
            continue

        stem = Path(file_name).stem
        log_path = file_log.switch_to(stem)

        logger.info(f"=== Processing {file_name} ===")
        if log_path:
            logger.info(f"Log:    {log_path}")

        try:
            _process_file(file_path, stem, args)
            logger.info(f"=== Finished {file_name} ===")
        except Exception as e:
            failed.append(file_name)
            logging.exception(f"Failed while processing {file_name}: {e}")

    file_log.close()

    if failed:
        raise SystemExit(
            f"error: {len(failed)} of {len(target_files)} file(s) failed: "
            + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
