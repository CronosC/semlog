import json
import pandas as pd
import os
import shutil
import re
import copy
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)


class DataPipeline:
    data = {
        "events": {
            "primary_key": "",
            "event_type_column": "",
            "timestamp_column": "",
            "attribute_columns": [],
            "file_path": "",
        },
        "objects": {},
        "relations": {
            "event_to_object": {},
            "object_to_object": {},
        },
    }

    def __init__(self, folderpath=None):
        """loads the data from the given folder, if fix indexing is set all id columns get their type as a prefix"""
        if folderpath != None:
            self.load(folderpath)

    #### DATA LOADING ####

    def load(self, folderpath):
        """loads the data for a given log into memory from a fixed folder structure"""
        logging.info("loading OCED-based data from %s..", folderpath)

        logging.info("   loading data description..")
        data_desc = self.load_data_desc(folderpath + "/data_description.json")

        logging.info("   loading events..")
        self.load_events(folderpath, data_desc["events"])
        logging.info("   loading objects..")
        self.load_objects(folderpath, data_desc["objects"])
        logging.info("   loading relations..")
        self.load_relations(folderpath, data_desc["relations"])
        logging.info("done." + "-" * 25)

    def load_data_desc(self, filepath):
        """loads a dictionary which describes the log folder structure for a given log"""
        with open(filepath, "r") as desc_file:
            data_desc = json.load(desc_file)

        return data_desc

    def load_events(self, folderpath, event_desc):
        """loads the events from the folder structure given a description of that folder structure as a dictionairy"""
        file_path = folderpath + "/" + event_desc["file_path"]
        primary_key = event_desc["primary_key"]

        df_events = pd.read_csv(file_path, index_col=primary_key)

        self.data["events"] = event_desc
        self.data["events"]["data"] = df_events

    def load_objects(self, folderpath, object_desc):
        """loads the objects from the folder structure given a description of that folder structure as a dictionairy"""
        for object_type in object_desc:
            file_path = folderpath + "/" + object_desc[object_type]["file_path"]
            primary_key = object_desc[object_type]["primary_key"]
            df_object = pd.read_csv(file_path, index_col=primary_key)

            self.data["objects"][object_type] = object_desc[object_type]
            self.data["objects"][object_type]["data"] = df_object

    def load_relations(self, folderpath, rel_desc):
        """loads both event to object and object to object relations from the folder structure given a description of
        that folder structure as a dictionairy"""
        for eo_rel_type in rel_desc["event_to_object"]:
            file_path = (
                folderpath + "/" + rel_desc["event_to_object"][eo_rel_type]["file_path"]
            )
            primary_key = rel_desc["event_to_object"][eo_rel_type]["primary_key"]
            df_eo_rel = pd.read_csv(file_path, index_col=primary_key)

            self.data["relations"]["event_to_object"][eo_rel_type] = rel_desc[
                "event_to_object"
            ][eo_rel_type]
            self.data["relations"]["event_to_object"][eo_rel_type]["data"] = df_eo_rel

        for oo_rel_type in rel_desc["object_to_object"]:
            file_path = (
                folderpath
                + "/"
                + rel_desc["object_to_object"][oo_rel_type]["file_path"]
            )
            primary_key = rel_desc["object_to_object"][oo_rel_type]["primary_key"]
            df_oo_rel = pd.read_csv(file_path, index_col=primary_key)

            self.data["relations"]["object_to_object"][oo_rel_type] = rel_desc[
                "object_to_object"
            ][oo_rel_type]
            self.data["relations"]["object_to_object"][oo_rel_type]["data"] = df_oo_rel

    #### DATA SAVING ####

    def save(self, folderpath, delete_old=True):
        """saves the data for a given log from memory into a fixed folder structure"""
        logging.info("saving OCED-based data to %s..", folderpath)
        if delete_old:
            logging.info("   deleting old files..")
            shutil.rmtree(folderpath)
        logging.info("   creating folder structure..")
        self.ensure_directories_for_saving(folderpath)

        logging.info("   saving events..")
        self.save_events(folderpath)
        logging.info("   saving objects..")
        self.save_objects(folderpath)
        logging.info("   saving relations..")
        self.save_relations(folderpath)

        logging.info("   saving data description..")
        self.save_data_description(folderpath)
        logging.info("done." + "-" * 25)

    def ensure_directories_for_saving(self, folderpath):
        """creates the directory structure for our dataformat"""
        events_path = os.path.join(folderpath, "events")
        objects_path = os.path.join(folderpath, "objects")
        eo_rel_path = os.path.join(folderpath, "relations/event_to_object")
        oo_rel_path = os.path.join(folderpath, "relations/object_to_object")

        os.makedirs(events_path, exist_ok=True)
        os.makedirs(objects_path, exist_ok=True)
        os.makedirs(eo_rel_path, exist_ok=True)
        os.makedirs(oo_rel_path, exist_ok=True)

    def save_data_description(self, folderpath):
        """first creates the data description via dropping the dataframes from the dictionariy in memory and then saves it as json"""
        data_desc = copy.deepcopy(self.data)
        data_desc["events"].pop("data", None)
        for object_type in data_desc["objects"]:
            data_desc["objects"][object_type].pop("data", None)
        for eo_rel_type in data_desc["relations"]["event_to_object"]:
            data_desc["relations"]["event_to_object"][eo_rel_type].pop("data", None)
        for oo_rel_type in data_desc["relations"]["object_to_object"]:
            data_desc["relations"]["object_to_object"][oo_rel_type].pop("data", None)

        desc_path = os.path.join(folderpath, "data_description.json")
        with open(desc_path, "w") as desc_file:
            json.dump(data_desc, desc_file)

    def save_events(self, folderpath):
        """saves the events data as a csv in a subfolder of the given path"""
        df_events = self.data["events"]["data"]
        full_events_path = os.path.join(folderpath, "events/events.csv")
        self.data["events"]["file_path"] = "events/events.csv"
        self.save_dataframe(df_events, full_events_path)

    def save_objects(self, folderpath):
        """saves the objects data as a csv in a subfolder of the given path"""
        for object_type in self.data["objects"]:
            df_object = self.data["objects"][object_type]["data"]

            object_sub_path = os.path.join("objects/", str(object_type) + ".csv")
            full_object_path = os.path.join(folderpath, object_sub_path)
            self.data["objects"][object_type]["file_path"] = object_sub_path

            self.save_dataframe(df_object, full_object_path)

    def save_relations(self, folderpath):
        """saves the relation data, both event to object and object to object, as a csv in a subfolder of the given path"""
        for eo_rel_type in self.data["relations"]["event_to_object"]:
            df_eo_rel = self.data["relations"]["event_to_object"][eo_rel_type]["data"]

            eo_rel_sub_path = os.path.join(
                "relations/event_to_object", str(eo_rel_type) + ".csv"
            )
            full_eo_rel_path = os.path.join(folderpath, eo_rel_sub_path)
            self.data["relations"]["event_to_object"][eo_rel_type][
                "file_path"
            ] = eo_rel_sub_path

            self.save_dataframe(df_eo_rel, full_eo_rel_path)
        for oo_rel_type in self.data["relations"]["object_to_object"]:
            df_oo_rel = self.data["relations"]["object_to_object"][oo_rel_type]["data"]

            oo_rel_sub_path = os.path.join(
                "relations/object_to_object", str(oo_rel_type) + ".csv"
            )
            full_oo_rel_path = os.path.join(folderpath, oo_rel_sub_path)
            self.data["relations"]["object_to_object"][oo_rel_type][
                "file_path"
            ] = oo_rel_sub_path

            self.save_dataframe(df_oo_rel, full_oo_rel_path)

    def save_dataframe(self, df, filepath):
        """writes a given pandas dataframe to a given path"""
        directory = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)
        df.to_csv(filepath)
