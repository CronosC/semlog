import json
import pandas as pd
import pm4py as pm
import os
import shutil
import re
import copy
import logging
import warnings

from semanticlog.enrich.declare import DeclareEnricher


class DataPipeline:
    data = {
        "events": {
            "primary_key": "",
            "event_type_column": "",
            "timestamp_column": "",
            "attribute_columns": [],
            "file_path": "",
            "case_concepts": {
                "df_x": {
                    "file_path": "",
                    "primary_key": "",
                    "event_key": "",
                    "timestamp_column": "",
                }
            },
        },
        "objects": {},
        "relations": {
            "event_to_object": {},
            "object_to_object": {},
        },
        "misc": {
            "case_concepts": {},
            "declare": {}
        }
    }

    def __init__(self, folderpath, load=True, ignore_declare=False):
        """loads the data from the given folder, if fix indexing is set all id columns get their type as a prefix"""
        self.folderpath = folderpath
        if load:
            self.load(ignore_declare=ignore_declare)

    #### DATA LOADING ####

    def load(self, ignore_declare=False):
        """loads the data for a given log into memory from a fixed folder structure"""
        logging.info("loading OCED-based data from %s..", self.folderpath)

        logging.info("   loading data description..")
        data_desc = self.load_data_desc()

        logging.info("   loading events..")
        self.load_events(data_desc["events"])
        logging.info("   loading objects..")
        self.load_objects(data_desc["objects"])
        logging.info("   loading relations..")
        self.load_relations(data_desc["relations"])
        
        if "misc" in data_desc.keys():
            if "case_concepts" in data_desc["misc"].keys():
                logging.info("   loading case concepts..")
                self.load_case_concepts(data_desc["misc"]["case_concepts"])
            if "declare" in data_desc["misc"].keys() and not ignore_declare:
                logging.info("   loading declare data..")
                self.load_declare(data_desc["misc"]["declare"])
        logging.info("done." + "-" * 25)

    def load_data_desc(self):
        """loads a dictionary which describes the log folder structure for a given log"""
        filepath = os.path.join(self.folderpath, "data_description.json")
        with open(filepath, "r") as desc_file:
            data_desc = json.load(desc_file)

        return data_desc

    def load_events(self, event_desc):
        """loads the events from the folder structure given a description of that folder structure as a dictionairy"""
        file_path = self.folderpath + "/" + event_desc["file_path"]
        primary_key = event_desc["primary_key"]

        df_events = pd.read_csv(file_path, index_col=primary_key)

        self.data["events"] = event_desc
        self.data["events"]["data"] = df_events

    def load_objects(self, object_desc):
        """loads the objects from the folder structure given a description of that folder structure as a dictionairy"""
        for object_type in object_desc:
            file_path = self.folderpath + "/" + object_desc[object_type]["file_path"]
            primary_key = object_desc[object_type]["primary_key"]
            df_object = pd.read_csv(file_path, index_col=primary_key)

            self.data["objects"][object_type] = object_desc[object_type]
            self.data["objects"][object_type]["data"] = df_object

    def load_relations(self, rel_desc):
        """loads both event to object and object to object relations from the folder structure given a description of
        that folder structure as a dictionairy"""
        for eo_rel_type in rel_desc["event_to_object"]:
            file_path = os.path.join(
                self.folderpath, rel_desc["event_to_object"][eo_rel_type]["file_path"]
            )
            primary_key = rel_desc["event_to_object"][eo_rel_type]["primary_key"]
            df_eo_rel = pd.read_csv(file_path, index_col=primary_key)

            self.data["relations"]["event_to_object"][eo_rel_type] = rel_desc[
                "event_to_object"
            ][eo_rel_type]
            self.data["relations"]["event_to_object"][eo_rel_type]["data"] = df_eo_rel

        for oo_rel_type in rel_desc["object_to_object"]:
            file_path = os.path.join(
                self.folderpath, rel_desc["object_to_object"][oo_rel_type]["file_path"]
            )
            primary_key = rel_desc["object_to_object"][oo_rel_type]["primary_key"]
            df_oo_rel = pd.read_csv(file_path, index_col=primary_key)

            self.data["relations"]["object_to_object"][oo_rel_type] = rel_desc[
                "object_to_object"
            ][oo_rel_type]
            self.data["relations"]["object_to_object"][oo_rel_type]["data"] = df_oo_rel
            
    def load_case_concepts(self, case_co_desc):
        """loads case concepts from the folder structure given a description of that folder structure as a dictionairy"""
        for case_co_name in case_co_desc:
            file_path = self.folderpath + "/" + case_co_desc[case_co_name]["file_path"]
            primary_key = case_co_desc[case_co_name]["primary_key"]

            df_case_co = pd.read_csv(file_path, index_col=primary_key)

            self.data["misc"]["case_concepts"][case_co_name] = case_co_desc[
                case_co_name
            ]
            self.data["misc"]["case_concepts"][case_co_name]["data"] = df_case_co
            
    def load_declare(self, declare_desc):
        """loads declare constraint data from the folder structure given a description of that folder structure as a dictionairy"""
        for declare_name in declare_desc:
            file_path = os.path.join(self.folderpath, declare_desc[declare_name]["file_path"])

            df_declare = pd.read_csv(file_path)

            self.data["misc"]["declare"][declare_name] = declare_desc[
                declare_name
            ]
            self.data["misc"]["declare"][declare_name]["data"] = df_declare

    #### DATA SAVING ####

    def save(self, delete_old=True):
        """saves the data for a given log from memory into a fixed folder structure"""
        logging.info("saving OCED-based data to %s..", self.folderpath)
        if delete_old:
            logging.info("   deleting old files..")
            try:
                shutil.rmtree(self.folderpath)
            except:
                logging.info("     no old files found..")
        logging.info("   creating folder structure..")
        self.ensure_directories_for_saving()

        logging.info("   saving events..")
        self.save_events()
        logging.info("   saving objects..")
        self.save_objects()
        logging.info("   saving relations..")
        self.save_relations()
        if "case_concepts" in self.data["misc"].keys():
            logging.info("   saving case concepts..")
            self.save_case_concepts()
        if "declare" in self.data["misc"].keys():
            logging.info("   saving declare constraints..")
            self.save_declare_constraints()

        logging.info("   saving data description..")
        self.save_data_description()
        logging.info("done." + "-" * 25)

    def ensure_directories_for_saving(self):
        """creates the directory structure for our dataformat"""
        events_path = os.path.join(self.folderpath, "events")
        objects_path = os.path.join(self.folderpath, "objects")
        eo_rel_path = os.path.join(self.folderpath, "relations/event_to_object")
        oo_rel_path = os.path.join(self.folderpath, "relations/object_to_object")
        case_co_path = os.path.join(self.folderpath, "misc/case_concepts")
        declare_path = os.path.join(self.folderpath, "misc/declare")

        os.makedirs(events_path, exist_ok=True)
        os.makedirs(objects_path, exist_ok=True)
        os.makedirs(eo_rel_path, exist_ok=True)
        os.makedirs(oo_rel_path, exist_ok=True)
        os.makedirs(case_co_path, exist_ok=True)
        os.makedirs(declare_path, exist_ok=True)

    def save_data_description(self):
        """first creates the data description via dropping the dataframes from the dictionariy in memory and then saves it as json"""
        data_desc = copy.deepcopy(self.data)
        data_desc["events"].pop("data", None)
        for object_type in data_desc["objects"]:
            data_desc["objects"][object_type].pop("data", None)
        for eo_rel_type in data_desc["relations"]["event_to_object"]:
            data_desc["relations"]["event_to_object"][eo_rel_type].pop("data", None)
        for oo_rel_type in data_desc["relations"]["object_to_object"]:
            data_desc["relations"]["object_to_object"][oo_rel_type].pop("data", None)
        if "case_concepts" in data_desc["misc"].keys():
            for case_co_name in data_desc["misc"]["case_concepts"]:
                data_desc["misc"]["case_concepts"][case_co_name].pop("data", None)
        if "declare" in data_desc["misc"].keys():
            for declare_constraints in data_desc["misc"]["declare"]:
                data_desc["misc"]["declare"][declare_constraints].pop("data", None)

        desc_path = os.path.join(self.folderpath, "data_description.json")
        with open(desc_path, "w") as desc_file:
            json.dump(data_desc, desc_file)

    def save_events(self):
        """saves the events data as a csv in a subfolder of the given path"""
        df_events = self.data["events"]["data"]
        full_events_path = os.path.join(self.folderpath, "events/events.csv")
        self.data["events"]["file_path"] = "events/events.csv"
        self.save_dataframe(df_events, full_events_path)

    def save_objects(self):
        """saves the objects data as a csv in a subfolder of the given path"""
        for object_type in self.data["objects"]:
            df_object = self.data["objects"][object_type]["data"]

            object_sub_path = os.path.join("objects/", str(object_type) + ".csv")
            full_object_path = os.path.join(self.folderpath, object_sub_path)
            self.data["objects"][object_type]["file_path"] = object_sub_path

            self.save_dataframe(df_object, full_object_path)

    def save_relations(self):
        """saves the relation data, both event to object and object to object, as a csv in a subfolder of the given path"""
        for eo_rel_type in self.data["relations"]["event_to_object"]:
            df_eo_rel = self.data["relations"]["event_to_object"][eo_rel_type]["data"]

            eo_rel_sub_path = os.path.join(
                "relations/event_to_object", str(eo_rel_type) + ".csv"
            )
            full_eo_rel_path = os.path.join(self.folderpath, eo_rel_sub_path)
            self.data["relations"]["event_to_object"][eo_rel_type][
                "file_path"
            ] = eo_rel_sub_path

            self.save_dataframe(df_eo_rel, full_eo_rel_path)
        for oo_rel_type in self.data["relations"]["object_to_object"]:
            df_oo_rel = self.data["relations"]["object_to_object"][oo_rel_type]["data"]

            oo_rel_sub_path = os.path.join(
                "relations/object_to_object", str(oo_rel_type) + ".csv"
            )
            full_oo_rel_path = os.path.join(self.folderpath, oo_rel_sub_path)
            self.data["relations"]["object_to_object"][oo_rel_type][
                "file_path"
            ] = oo_rel_sub_path

            self.save_dataframe(df_oo_rel, full_oo_rel_path)

    def save_case_concepts(self):
        """saves the case concepts data as a csv in a subfolder of the given path"""
        for case_co_name in self.data["misc"]["case_concepts"]:
            df_case_co = self.data["misc"]["case_concepts"][case_co_name]["data"]

            case_co_sub_path = os.path.join(
                "misc/case_concepts", str(case_co_name) + ".csv"
            )
            full_case_co_path = os.path.join(self.folderpath, case_co_sub_path)
            self.data["misc"]["case_concepts"][case_co_name][
                "file_path"
            ] = case_co_sub_path

            self.save_dataframe(df_case_co, full_case_co_path)

    def save_declare_constraints(self):
        """saves the case concepts data as a csv in a subfolder of the given path"""
        for declare_constraint_view in self.data["misc"]["declare"]:
            df_declare = self.data["misc"]["declare"][declare_constraint_view]["data"]

            declare_sub_path = os.path.join(
                "misc/declare", str(declare_constraint_view) + ".csv"
            )
            full_declare_path = os.path.join(self.folderpath, declare_sub_path)
            self.data["misc"]["declare"][declare_constraint_view]["file_path"] = declare_sub_path

            self.save_dataframe(df_declare, full_declare_path)

    def add_all_declare_constraints(self, declare_constraints, consider_vacuity=False):
        """adds the declare constraints at the trace level to all case_concepts in the data"""
        for case_concept_name in self.data["misc"]["case_concepts"]:
            self.add_declare_constraints(case_concept_name, declare_constraints)

    def add_declare_constraints(
        self, case_concept_name, declare_constraints, consider_vacuity=False
    ):
        """constructs a declare constraint file via the DeclareEnricher object,
        the declare_constraints must be a pair of strings corresponding to unary
        and binary declare constraints"""
        enricher = self.prepare_declare_enricher(case_concept_name)
        

        declare_name = "declare_" + enricher.case_column
        declare_model_path = os.path.join(
            self.folderpath, "misc/declare/models", "model_" + enricher.case_column + ".decl"
        )
        enricher.construct_declare_model(
            declare_constraints[0],
            declare_constraints[1],
            declare_model_path,
        )

        enricher.evaluate_declare_model(declare_model_path, consider_vacuity=consider_vacuity)

        trace_constraints_mapping = enricher.results.get_metric(metric="state")
        trace_constraints_mapping = trace_constraints_mapping.rename(columns=self.preprocess_constraint_string)
        trace_constraints_mapping[enricher.case_column] = (
            trace_constraints_mapping.index.map(
                lambda x: enricher.log.get_trace(x).attributes["concept:name"]
            )
        )

        declare_sub_path = os.path.join(
            "misc/declare", "declare_" + enricher.case_column + ".csv"
        )
        full_declare_path = os.path.join(self.folderpath, declare_sub_path)

        if "misc" not in self.data.keys():
            self.data["misc"] = {}
        if "declare" not in self.data["misc"].keys():
            self.data["misc"]["declare"] = {}
        if declare_name not in self.data["misc"]["declare"].keys():
            self.data["misc"]["declare"][declare_name] = {}
        self.data["misc"]["declare"][declare_name]["file_path"] = declare_sub_path
        self.data["misc"]["declare"][declare_name]["df_view"] = enricher.case_column
        self.data["misc"]["declare"][declare_name]["data"] = trace_constraints_mapping
        self.save()
        
    def preprocess_constraint_string(self, constraint_string):
        constraint_string = constraint_string.replace("|", "")
        constraint_string = constraint_string.replace(" ", "")
        constraint_string = constraint_string.replace("[", "(")
        constraint_string = constraint_string.replace("]", ")")
        return constraint_string

    def prepare_declare_enricher(self, case_concept_name):
        """sets up and returns an DeclareEnricher object with the given data"""
        os.makedirs(os.path.join(self.folderpath, "misc", "declare", "models"), exist_ok=True)
        case_concept_log_path = os.path.join(
            self.folderpath,
            self.data["misc"]["case_concepts"][case_concept_name]["file_path"],
        )

        activity_column = self.data["misc"]["case_concepts"][case_concept_name][
            "event_type_column"
        ]
        time_column = self.data["misc"]["case_concepts"][case_concept_name][
            "timestamp_column"
        ]
        case_column = self.data["misc"]["case_concepts"][case_concept_name][
            "primary_key"
        ]

        log = pd.read_csv(case_concept_log_path)
        log = pm.format_dataframe(
            log,
            case_id=case_column,
            activity_key=activity_column,
            timestamp_key=time_column,
        )
        log = pm.convert_to_event_log(log)

        enricher = DeclareEnricher(
            log,
            event_type_column=activity_column,
            case_column=case_column,
            time_column=time_column,
        )

        return enricher

    def save_dataframe(self, df, filepath):
        """writes a given pandas dataframe to a given path"""
        directory = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)
        df.to_csv(filepath)
