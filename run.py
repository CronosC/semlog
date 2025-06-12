from semanticlog.data.classicalloader import ClassicalLoader
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.settings import OntoBuildSettings, EventOrderRepresentation, OwlProfile
from semanticlog.data.cleaning import clean_oced_data

import logging

logging.basicConfig(level=logging.INFO)

cl = ClassicalLoader("data/au_timestamp_start.csv", {
    "case_col": "video_id",
    "act_col": "active_au",
    "time_col": "timestamp_start"
})

print("!")

oced = cl.get_oced()
oced = clean_oced_data(oced)

print("!")

settings = OntoBuildSettings()
settings.profile = OwlProfile.OWL2DL
settings.order_representation = EventOrderRepresentation.DECLARE
settings.declare_file = "data/emotions_declare.csv"

print("!")

builder = OCEDOntoBuilder("https:/oced", settings)
builder.build_ontology(oced)
builder.save_onto("data/tmp.owl")