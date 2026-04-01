import logging

# --- Logging setup ---
import logging.config
import os
import random
from datetime import datetime, timedelta

from pm4py.algo.simulation.playout.process_tree import algorithm as pt_playout
from pm4py.algo.simulation.tree_generator.variants import basic
from pm4py.objects.conversion.log import converter as log_converter

from semanticlog.data.classicalloader import ClassicalLoader
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.misc import quantize
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    EventOrderRepresentation,
    OntoBuildSettings,
    OwlProfile,
)
from semanticlog.onto.upper import OCEDUpperOntoBuilder

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

if os.path.exists("data/temp/tmp.owl"):
    try:
        os.remove("data/temp/tmp.owl")
        logger.info("Removed temporary ontology file: data/temp/tmp.owl")
    except Exception:
        logger.exception("Failed to remove data/temp/tmp.owl")


N = 1000
log_path = "data/misc/simulated_log.csv"

# generate a random process tree
pt = basic.apply({"max_rec_depth": 3, "prob_leaf": 0.3})
print(pt)

# play out to a log
log = pt_playout.apply(pt, parameters={"no_traces": N})
# 2) ensure case ids + timestamps
for i, trace in enumerate(log):
    t = datetime(2025, 1, 1)
    trace.attributes["concept:name"] = str(i)  # case id on trace
    for ev in trace:
        t += timedelta(minutes=random.randint(1, 10))
        ev["case:concept:name"] = str(i)
        ev["time:timestamp"] = t
# make df
log = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

resource_role_1 = ["res_" + chr(i) for i in range(ord("a"), ord("z") + 1)]
resource_role_2 = ["res_" + chr(i) for i in range(ord("A"), ord("J") + 1)]
n_activities = len(log["concept:name"].unique())
activities_role_1 = random.sample(
    log["concept:name"].unique().tolist(), int(n_activities * 0.75)
)

for activity in log["concept:name"].unique():
    log["resource"] = log["concept:name"].apply(
        lambda v: (
            random.choice(resource_role_1)
            if v in activities_role_1
            else random.choice(resource_role_2)
        )
    )

log.to_csv(log_path)

# check to ensure intended structure
actual_activities_role_1 = log[log["resource"].isin(resource_role_1)][
    "concept:name"
].unique()
actual_activities_role_2 = log[~log["resource"].isin(resource_role_1)][
    "concept:name"
].unique()
print(actual_activities_role_1, actual_activities_role_2)
assert set(actual_activities_role_1).intersection(actual_activities_role_2) == set()


iri = DEFAULT_IRI
upper_path = "data/onto/Upper.owl"
onto_path = "data/onto/Simulated_Quantized_OWL2DL_DF.owl"
materialize = False

logger.info(f"Input OCEL: {log_path}")
logger.info(f"Upper ontology: {upper_path}")
logger.info(f"Output ontology: {onto_path}")

logger.info("Loading OCEL data..")
loader = ClassicalLoader(log_path, {"object_cols": ["resource"]})
oced = loader.get_oced()
logger.info("Cleaning OCED data..")
oced = clean_oced_data(oced)
oced = quantize(oced)

settings = OntoBuildSettings()
settings.profile = OwlProfile.OWL2DL
settings.order_representation = [EventOrderRepresentation.DF]
settings.case_concepts = ["case_concept_name", "resource"]

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
