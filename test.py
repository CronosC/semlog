import os
import logging
import pandas as pd
from pandas.testing import assert_frame_equal
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.upper import OCEDUpperOntoBuilder
from semanticlog.data.pm4pyloader import PM4PYLoader
from semanticlog.data.ocelloader import OCELLoader
from semanticlog.data.cleaning import clean_oced_data, null_check
from semanticlog.data.filtering import filter_variants_by_maximum_repetition
from semanticlog.onto.settings import (
    OntoBuildSettings,
    OwlProfile,
    EventOrderRepresentation,
)
from semanticlog.onto.reason import Reasoner
from semanticlog.data.misc import quantize

# --- Logging setup ---
import logging.config

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)
logging.root.handlers = []
logging.basicConfig(
    format="%(name)s | %(asctime)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("oced.pipeline")

if os.path.exists("data/tmp.owl"):
    try:
        os.remove("data/tmp.owl")
        logger.info("Removed temporary ontology file: data/tmp.owl")
    except Exception:
        logger.exception("Failed to remove data/tmp.owl")

folder_path = "data/OCEL"
files = [
    "Hospital.xml",
]

ccs = {}

for file_name in os.listdir(folder_path):
    settings = OntoBuildSettings()
    settings.profile = OwlProfile.OWL2DL
    settings.order_representation = [EventOrderRepresentation.DF]
    settings.inverse_properties = True
    settings.make_inverses_explicit = True
    materialize = False
    quantize_data = False

    if any(s in file_name for s in files):

        iri = "https://oced#"
        upper_path = "data/ONTO/Upper.owl"
        or_str = ""
        for or_ in settings.order_representation:
            or_str = or_str + "" + str(or_).split(".")[1]
        mat_str = ""
        if materialize:
            mat_str = "_Mat"
        quant_str = ""
        if quantize_data:
            quant_str = "_Quantized"
        onto_path = (
            "data/ONTO/"
            + file_name.split(".")[0]
            + mat_str
            + quant_str
            + "_"
            + str(settings.profile).split(".")[1]
            + "_"
            + str(or_str)
            + ".owl"
        )
        file_path = os.path.join(folder_path, file_name)

        logger.info(f"=== Processing {file_name} ===")
        logger.info(f"Input OCEL: {file_path}")
        logger.info(f"Upper ontology: {upper_path}")
        logger.info(f"Output ontology: {onto_path}")

        try:
            logger.info("Loading OCEL data..")
            loader = PM4PYLoader(file_path, {})
            oced = loader.get_oced()
            logger.info("Cleaning OCED data..")
            oced = clean_oced_data(oced)
            if quantize_data:
                oced = quantize(oced)
            #contains_null_values = null_check(oced)
            #assert not contains_null_values

            if ccs.get(file_name) is None:
                logger.info("Selecting most common objects as case concepts (n=3)..")
                ccs[file_name] = oced.get_most_common_objects(n=3)
            settings.case_concepts = ccs[file_name]
            logger.info(f"Case concepts: {settings.case_concepts}")

            logger.info("Building upper ontology..")
            builder = OCEDUpperOntoBuilder(iri, settings)
            builder.construct_upper_ontology(upper_path)

            logger.info("Instantiating OCEL data into ontology..")
            builder = OCEDOntoBuilder(upper_path, settings)
            builder.build_ontology(oced)
            builder.save_onto(onto_path)
            logger.info(f"Ontology saved: {onto_path}")

            if materialize:
                logger.info("Materializing with reasoner..")
                reasoner = Reasoner(onto_path)
                reasoner.materialize()
                reasoner.save_onto()
                logger.info("Materialization complete and saved.")

            logger.info(f"=== Finished {file_name} ===")
        except Exception as e:
            logger.exception(f"Failed while processing {file_name}: {e}")
