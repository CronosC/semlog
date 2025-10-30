import os
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.upper import OCEDUpperOntoBuilder
from semanticlog.data.pm4pyloader import PM4PYLoader
from semanticlog.data.ocelloader import OCELLoader
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.filtering import filter_variants_by_maximum_repetition
from semanticlog.onto.settings import OntoBuildSettings, OwlProfile, EventOrderRepresentation

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})
logging.root.handlers = []
logging.basicConfig(
    format="%(name)s | %(asctime)s | %(message)s", level=logging.INFO, datefmt="%H:%M:%S"
)

class TestLoadAndTransformOCEL(unittest.TestCase):
    
    def test_load_and_transform_all(self):
        if os.path.exists("data/tmp.owl"):
            os.remove("data/tmp.owl")
            
        size_limit_mb = 100
        size_limit_bytes = size_limit_mb * 1024 * 1024
        
        folder_path = "data/OCEL"
        for file_name in os.listdir(folder_path):
            if file_name not in ["AOE2.sqlite", "AOE2_small.json", "ContainerLogistics.json"]:
                file_path = os.path.join(folder_path, file_name)
                file_size = os.path.getsize(file_path)
                if file_size < size_limit_bytes:
                    with self.subTest(file=file_name): 
                        settings = OntoBuildSettings()
                        settings.profile = OwlProfile.OWL2EL
                        settings.order_representation = []
                        settings.case_concepts = []
                        builder = OCEDUpperOntoBuilder("https://oced#", settings)
                        builder.construct_upper_ontology("data/ONTO/Upper.owl")
                        file_path = os.path.join(folder_path, file_name)
                        loader = PM4PYLoader(file_path, {})
                        oced = loader.get_oced()
                        oced = clean_oced_data(oced)
   
                        builder = OCEDOntoBuilder("data/ONTO/Upper.owl", settings)
                        builder.build_ontology(oced)
                        builder.save_onto("data/ONTO/" + file_name.split(".")[0] + ".owl")
                        
              
    def test_p2p_log(self):
        folder_path = "data/OCEL"
        for file_name in os.listdir(folder_path):
            if file_name in ["P2P.xml"]:
                file_path = os.path.join(folder_path, file_name)
                file_size = os.path.getsize(file_path)
                settings = OntoBuildSettings()
                settings.profile = OwlProfile.OWL2EL
                settings.order_representation = []
                settings.case_concepts = []
                builder = OCEDUpperOntoBuilder("https://oced#", settings)
                builder.construct_upper_ontology("data/ONTO/Upper.owl")
                file_path = os.path.join(folder_path, file_name)
                loader = PM4PYLoader(file_path, {})
                oced = loader.get_oced()
                oced = clean_oced_data(oced)
                builder = OCEDOntoBuilder("data/ONTO/Upper.owl", settings)
                builder.build_ontology(oced)
                builder.save_onto("data/ONTO/" + file_name.split(".")[0] + ".owl")
'''
class TestLoadAndTransformOCEL(unittest.TestCase):
    
    def setUp(self):
        if os.path.exists("data/tmp.owl"):
            os.remove("data/tmp.owl")
        builder = OCEDUpperOntoBuilder("https://oced#")
        builder.construct_upper_ontology("data/tmp.owl")

    def test_load_and_transform_ocel_sqlite_github_pm4py(self):
        loader = PM4PYLoader("data/github_pm4py.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        oced = clean_oced_data(oced)
        builder = OCEDOntoBuilder("data/tmp.owl")
        builder.build_ontology(oced)
        builder.save_onto("data/tmp.owl")
        

    def test_load_and_transform_ocel_sqlite_o2c(self):
        loader = PM4PYLoader("data/o2c.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        oced = clean_oced_data(oced)
        builder = OCEDOntoBuilder("data/tmp.owl")
        builder.build_ontology(oced)
        builder.save_onto("data/tmp.owl")
        
    def test_load_and_transform_ocel_sqlite_ocel2_p2p(self):
        loader = PM4PYLoader("data/ocel2-p2p.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        oced = clean_oced_data(oced)
        builder = OCEDOntoBuilder("data/tmp.owl")
        builder.build_ontology(oced)
        builder.save_onto("data/tmp.owl")

    def test_load_and_transform_ocel_sqlite_order_management(self):
        loader = PM4PYLoader("data/order-management.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        oced = clean_oced_data(oced)
        builder = OCEDOntoBuilder("data/tmp.owl")
        builder.build_ontology(oced)
        builder.save_onto("data/tmp.owl")
        
    def test_load_and_transform_ocel_sqlite_p2p(self):
        loader = PM4PYLoader("data/p2p.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        oced = clean_oced_data(oced)
        print(oced.get_activities())
        print("!"*100)
        print(oced.get_object_types())
        print("!"*100)
        print(oced.get_relation_types())
        print("!"*100)
        print(oced.get_object("MBLNR_ZEILE"))
        print(oced.get_object("MBLNR_ZEILE")["data"]["matnr_attr"].unique())
        print(len(oced.get_object("MBLNR_ZEILE")["data"]["matnr_attr"].unique()))
        builder = OCEDOntoBuilder("data/tmp.owl")
        builder.build_ontology(oced)
        builder.save_onto("data/tmp.owl")
    
    def test_load_and_transform_ocel_sqlite_recruiting(self):
        loader = PM4PYLoader("data/recruiting.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        oced = clean_oced_data(oced)
        builder = OCEDOntoBuilder("data/tmp.owl")
        builder.build_ontology(oced)
        builder.save_onto("data/tmp.owl")


    def test_load_and_transform_ocel_sqlite_running_example(self):
        loader = PM4PYLoader("data/running-example.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        oced = clean_oced_data(oced)
        builder = OCEDOntoBuilder("data/tmp.owl")
        builder.build_ontology(oced)
        builder.save_onto("data/tmp.owl")

    def test_load_and_transform_ocel_sqlite_windows_events(self):
        loader = PM4PYLoader("data/windows_events.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        oced = clean_oced_data(oced)
        builder = OCEDOntoBuilder("data/tmp.owl")
        builder.build_ontology(oced)
        builder.save_onto("data/tmp.owl")

    def test_load_and_transform_ocel_sqlite_transfer_order(self):
        loader = PM4PYLoader("data/transfer_order.sqlite", {"file_format": "sqlite"})
        oced = loader.get_oced()
        oced = clean_oced_data(oced)
        builder = OCEDOntoBuilder("data/tmp.owl")
        builder.build_ontology(oced)
        builder.save_onto("data/tmp.owl")
        
'''

if __name__ == "__main__":
    unittest.main()
    
