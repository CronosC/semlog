import argparse
import glob
import logging
import logging.config
import os
from datetime import datetime

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.misc import quantize
from semanticlog.data.pm4pyloader import PM4PYLoader
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    DataRepresentation,
    EventOrderRepresentation,
    OcelExtension,
    OntoBuildSettings,
    OwlProfile,
)
from semanticlog.onto.upper import OCEDUpperOntoBuilder

LOG_KEEP = 10
_LOG_FORMAT = "%(name)s | %(asctime)s | %(message)s"
_LOG_DATEFMT = "%H:%M:%S"
_file_handler: logging.FileHandler | None = None

logger = logging.getLogger("oced.pipeline")


def _setup_console_logging() -> None:
    logging.config.dictConfig({"version": 1, "disable_existing_loggers": True})
    logging.root.handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)


def _set_file_log(stem: str, log_dir: str) -> str:
    global _file_handler
    os.makedirs(log_dir, exist_ok=True)

    if _file_handler is not None:
        logging.root.removeHandler(_file_handler)
        _file_handler.close()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{ts}_{stem}.log")
    _file_handler = logging.FileHandler(log_path)
    _file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    logging.root.addHandler(_file_handler)

    existing = sorted(glob.glob(os.path.join(log_dir, f"*_{stem}.log")))
    for old in existing[:-LOG_KEEP]:
        os.remove(old)

    return log_path


_PROFILE_MAP = {
    "OWL2DL": OwlProfile.OWL2DL,
    "OWL2EL": OwlProfile.OWL2EL,
}
_ORDER_MAP = {
    "DF": EventOrderRepresentation.DF,
    "EF": EventOrderRepresentation.EF,
    "EF_explicit": EventOrderRepresentation.EF_explicit,
    "DECLARE": EventOrderRepresentation.DECLARE,
}
_EXTENSION_MAP = {
    "ATTRIBUTE_HISTORY": OcelExtension.ATTRIBUTE_HISTORY,
    "RELATION_TIMESTAMPS": OcelExtension.RELATION_TIMESTAMPS,
}
_DATA_REPR_MAP = {
    "none": DataRepresentation.NONE,
    "values": DataRepresentation.VALUES,
    "quantize_simple": DataRepresentation.QUANTIZE_SIMPLE,
    "quantize_dist": DataRepresentation.QUANTIZE_DIST,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OWL ontologies from OCEL event log files."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to folder containing OCEL input files.",
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
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bin numeric attributes (default: on).",
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
        choices=["ATTRIBUTE_HISTORY", "RELATION_TIMESTAMPS"],
        default=[],
        metavar="EXT",
        help=(
            "OCEL 2.0 extensions to enable. Choices: ATTRIBUTE_HISTORY "
            "(object attribute change history), RELATION_TIMESTAMPS "
            "(timed e2o relation observations). Default: none."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default="",
        help="Directory for file logs (default: '' = no file logging).",
    )
    parser.add_argument(
        "--data-representation",
        choices=["none", "values", "quantize_simple", "quantize_dist"],
        default="values",
        help="How numeric attributes are represented in the ontology (default: values).",
    )
    parser.add_argument(
        "--quantize-bins",
        type=int,
        default=5,
        metavar="N",
        help="Number of bins for quantize_simple/quantize_dist modes (default: 5).",
    )
    parser.add_argument(
        "--quantize-labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help=(
            "Bin labels for quantize modes. Must have exactly --quantize-bins elements. "
            "Defaults to very_low low average high very_high when --quantize-bins is 5."
        ),
    )
    parser.add_argument(
        "--inverse-properties",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Declare inverse properties in the ontology (default: on).",
    )
    parser.add_argument(
        "--make-inverses-explicit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Materialise inverse relation assertions in the ABox (default: on).",
    )
    parser.add_argument(
        "--case-concepts",
        nargs="*",
        default=None,
        metavar="CONCEPT",
        help="Case concept names to use. Overrides auto-selection from object frequency.",
    )
    return parser.parse_args()


def build_output_path(
    output_dir: str,
    stem: str,
    profile: str,
    order: list[str],
    quantize: bool,
    materialize: bool,
    extensions: list[str],
) -> str:
    mat_str = "_Mat" if materialize else ""
    quant_str = "_Quantized" if quantize else ""
    order_str = "_".join(order) if order else "NoOrder"
    ext_str = "_" + "_".join(extensions) if extensions else ""
    return os.path.join(
        output_dir, f"{stem}{mat_str}{quant_str}_{profile}_{order_str}{ext_str}.owl"
    )


def main() -> None:
    _setup_console_logging()
    args = parse_args()

    profile = _PROFILE_MAP[args.profile]
    order_repr = [_ORDER_MAP[o] for o in (args.order or [])]
    ocel_extensions = [_EXTENSION_MAP[e] for e in (args.ocel_extensions or [])]
    data_repr = _DATA_REPR_MAP[args.data_representation]
    size_limit_bytes = args.size_limit * 1024 * 1024 if args.size_limit > 0 else 0

    all_files = os.listdir(args.folder)
    target_files = (
        [f for f in all_files if f in args.files] if args.files else all_files
    )

    upper_path = os.path.join(args.output, "Upper.owl")
    ccs: dict = {}

    for file_name in target_files:
        file_path = os.path.join(args.folder, file_name)

        if size_limit_bytes and os.path.getsize(file_path) > size_limit_bytes:
            logger.info(f"Skipping {file_name} (exceeds {args.size_limit} MB limit)")
            continue

        stem = file_name.split(".")[0]
        log_path = _set_file_log(stem, args.log_dir) if args.log_dir else None

        onto_path = build_output_path(
            args.output,
            stem,
            args.profile,
            args.order or [],
            args.quantize,
            args.materialize,
            args.ocel_extensions or [],
        )

        logger.info(f"=== Processing {file_name} ===")
        if log_path:
            logger.info(f"Log:    {log_path}")
        logger.info(f"Input:  {file_path}")
        logger.info(f"Output: {onto_path}")

        try:
            logger.info("Loading OCEL data..")
            oced = PM4PYLoader(
                file_path, {}, ocel_extensions=ocel_extensions
            ).get_oced()
            logger.info("Cleaning OCED data..")
            oced = clean_oced_data(oced)
            if args.quantize and data_repr not in (
                DataRepresentation.QUANTIZE_SIMPLE, DataRepresentation.QUANTIZE_DIST
            ):
                oced = quantize(oced)

            if ccs.get(file_name) is None:
                if args.case_concepts is not None:
                    ccs[file_name] = args.case_concepts
                else:
                    logger.info("Selecting most common objects as case concepts (n=3)..")
                    ccs[file_name] = oced.get_most_common_objects(n=3)

            settings = OntoBuildSettings()
            settings.profile = profile
            settings.order_representation = order_repr
            settings.ocel_extensions = ocel_extensions
            settings.inverse_properties = args.inverse_properties
            settings.make_inverses_explicit = args.make_inverses_explicit
            settings.case_concepts = ccs[file_name]
            settings.data_representation = data_repr
            settings.quantize_bins = args.quantize_bins
            if args.quantize_labels is not None:
                settings.quantize_labels = args.quantize_labels
            logger.info(f"Case concepts: {settings.case_concepts}")

            logger.info("Building upper ontology..")
            OCEDUpperOntoBuilder(DEFAULT_IRI, settings).construct_upper_ontology(
                upper_path
            )

            logger.info("Instantiating OCEL data into ontology..")
            builder = OCEDOntoBuilder(upper_path, settings)
            builder.build_ontology(oced)
            builder.save_onto(onto_path)
            logger.info(f"Ontology saved: {onto_path}")

            if args.materialize:
                logger.info("Materializing with reasoner..")
                reasoner = Reasoner(onto_path, profile=profile)
                reasoner.materialize()
                logger.info("Materialization complete.")

            logger.info(f"=== Finished {file_name} ===")
        except Exception as e:
            logging.exception(f"Failed while processing {file_name}: {e}")
            raise


if __name__ == "__main__":
    main()
