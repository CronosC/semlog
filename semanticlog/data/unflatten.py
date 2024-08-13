import pandas as pd
import json
import os
import pm4py as pm
from pm4py.algo.filtering.log.variants import variants_filter
import re
import time
import random
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(format='%(asctime)s | %(message)s', level=logging.INFO)
random.seed(1234)


class XES_Preprocessor:

    def __init__(self):
        pass

    #### Cleaning ####

    def clean_dataframe(self, df):
        """applies string cleaning to all column names and row cleaning to all entries in the dataframe"""
        clean_df = df.map(self.clean_string_row)
        clean_columns = {
            col_str: self.clean_string(col_str) for col_str in clean_df.columns
        }
        clean_df = clean_df.rename(columns=clean_columns)
        return clean_df

    def clean_string(self, string):
        """replaces all special characters with underscores and cleans capizalization"""
        string = re.sub(r"[^A-Za-z0-9]", "_", string)
        string = string.casefold()
        return string

    def clean_string_row(self, row):
        """applies string cleaning to a row entry in a pandas dataframe, depending on its type"""
        if isinstance(row, str):
            return self.clean_string(row)
        return row
        
    def clean_column_values(self, log, to_clean):
        """gets a dictionary of columns and functions, applies the functions to the values in the corresponding columns"""
        for column, function in to_clean.items():
            if column in log.columns:
                log[column] = log[column].apply(function)
            else:
                raise ValueError(f"Column '{column}' not found in log DataFrame")
        return log

    #### Filtering ####

    def filter_to_timeintervall(self, log, mintime, maxtime, time_col="time:timestamp"):
        """takes a log, time intervall defined via a mintime and maxtime in unixtime seconds and returns the log filltered to this interval"""
        maxtime = pd.to_datetime(maxtime, unit="s", utc=True)
        mintime = pd.to_datetime(mintime, unit="s", utc=True)
        traces_outside_intervall = log[
            ((log["time:timestamp"] > maxtime) | (log["time:timestamp"] < mintime))
        ]
        log = log.drop(traces_outside_intervall.index)
        return log

    def filter_remove_months(
        self, log, months, time_col="time:timestamp", case_col="case:concept:name"
    ):
        """given a log and list of ints representing a month, filters the by removing any cases which have events in the given months"""
        events_in_months_to_remove = log[log[time_col].dt.month.isin(months)]
        case_ids_to_remove = events_in_months_to_remove[case_col].unique()
        log = log[~log[case_col].isin(case_ids_to_remove)]
        return log

    def filter_variants_by_minimum_occurence(self, log, minimum_occurences):
        """given a log and minimum number of occurences that variants should have, filter all variants with less occurences"""
        variants = variants_filter.get_variants(log)
        filtered_variants = {
            variant: cases
            for variant, cases in variants.items()
            if len(cases) >= minimum_occurences
        }
        log = variants_filter.apply(log, filtered_variants)
        return pm.convert_to_dataframe(log)

    def filter_variants_by_maximum_repetition(
        self, log, maximum_repetition, case_col="case:concept:name"
    ):
        """given a log and maximum number of allowed repetitions for a variant, filters out traces for each variant s.t. it is
        within the maximum number of allowed repetitions"""
        variants = variants_filter.get_variants(log)

        cases_to_keep = []
        for variant, cases in variants.items():
            if len(cases) > maximum_repetition:
                variant_cases_to_keep = random.sample(cases, maximum_repetition)
            else:
                variant_cases_to_keep = cases
            cases_to_keep = cases_to_keep + variant_cases_to_keep

        case_col_dict_structure = case_col.removeprefix("case:")

        case_ids_to_keep = [
            x.attributes[case_col_dict_structure] for x in cases_to_keep
        ]

        log = log[log[case_col].isin(case_ids_to_keep)]

        return pm.convert_to_dataframe(log)

    #### Misc. preprocessing ####

    def add_prefix(self, log, prefix_dict):
        """takes a log and a dict of columns, prefix pairs and returns a log where each column in the dict has the corresponding prefix"""
        for col, prefix in prefix_dict.items():
            if col in log.columns:
                log[col] = log[col].apply(lambda x: f"{prefix}{str(x)}")
        return log


class Unflattener_BPIC19:

    data = {
        "events": {},
        "objects": {},
        "relations": {
            "event_to_object": {},
            "object_to_object": {},
        },
    }

    def __init__(
        self,
        path,
        months_to_remove=[1, 12],
        max_variant_repetitions=10,
        min_traces_per_variant=2,
    ):
        logging.info("Constructing BPIC19 dataset from file %s..", path)
        self.PrePro = XES_Preprocessor()
        logging.info("   loading classical event log data..")
        log = self.load_raw_log(path)
        logging.info("   filtering data..")
        log = self.filter_log(
            log, months_to_remove, max_variant_repetitions, min_traces_per_variant
        )
        logging.info("   cleaning data..")
        log = self.clean_log(log)
        logging.info("   splitting data into entities and relations..")
        data = self.split_log(log)
        logging.info("done." + "-"*25)
        self.log = log
        self.data = data

    def load_raw_log(self, path):
        """loads a event log from a given path"""
        if path.endswith(".xes"):
            return pm.read_xes(path)
        else:
            raise ValueError("Can only read .xes logs, but was given the path:", path)

    def filter_log(
        self, log, months_to_remove, max_variant_repetitions, min_traces_per_variant
    ):
        """reduces the size of the BPIC log, removing outliers, variant repetitions and rare variants"""
        logging.info("      removing events before %i and after %i..", 1514592000, 1546300800)
        log = self.PrePro.filter_to_timeintervall(log, 1514592000, 1546300800)
        logging.info("      removing cases which have events in months corresponding to %s..", months_to_remove)
        log = self.PrePro.filter_remove_months(log, months_to_remove)
        logging.info("      removing case variants which occur less than %i times..", min_traces_per_variant)
        log = self.PrePro.filter_variants_by_minimum_occurence(
            log, min_traces_per_variant
        )
        logging.info("      limiting case variant repetitions to a maximum of %i times..", max_variant_repetitions)
        log = self.PrePro.filter_variants_by_maximum_repetition(
            log, max_variant_repetitions
        )
        return log

    def clean_log(self, log):
        """cleans up column names, renames columns, adds new key columns and adds index to key columns which are not specific enough"""
        log["event"] = log.index
        log["invoice"] = log.index

        logging.info("      renaming columns..")
        log = log.rename(
            columns={
                "org:resource": "resource",
                "concept:name": "activity",
                "Cumulative net worth (EUR)": "item_value",
                "time:timestamp": "timestamp",
                "case:Spend area text": "purchase_area",
                "case:Company": "company",
                "case:Document Type": "purchase_order_doc_type",
                "case:Sub spend area text": "subpurchase_area",
                "case:Purchasing Document": "purchase_order",
                "case:Vendor": "vendor",
                "case:Item Type": "item_type",
                "case:Item Category": "item_category",
                "case:Spend classification text": "spend_classification",
                "case:Name": "vendor_name",
                "case:GR-Based Inv. Verif.": "has_invoice_after_goods_receipt",
                "case:Item": "item",
                "case:concept:name": "purchase_order_item",
                "case:Goods Receipt": "has_goods_receipt",
            }
        )
        
        logging.info("      adding prefixes to key columns..")
        log = self.PrePro.add_prefix(
            log,
            {
                "event": "event_",
                "invoice": "invoice_",
                "item": "item_",
                "purchase_order_item": "purchase_order_item_",
                "purchase_order": "purchase_order_",
            },
        )
        logging.info("      cleaning special characters and unifying string casing..")
        log = self.PrePro.clean_dataframe(log)
        logging.info("      renaming some column values..")
        columns_to_clean = {
            "company": (lambda x : x.replace("companyid", "company")),
            "vendor": (lambda x : x.replace("vendorid", "vendor")),
            "vendor_name": (lambda x : x.replace("vendor_", "vendor_name_")),
        }
        log = self.PrePro.clean_column_values(log, columns_to_clean)
        return log

    def split_log(self, log):
        """splits the log into individual entities and relations"""
        
        events = {}
        objects = {}
        event_to_object = {}
        object_to_object = {}
        #### Events ####
        logging.info("      creating events..")
        
        events["data"] = log[["event", "activity", "timestamp"]]
        events["primary_key"] = "event"
        events["event_type_column"] = "activity"
        events["timestamp_column"] = "timestamp"
        events["attribute_columns"] = []
        events["file_path"] = ""
        
        logging.info("         created event data with a total of %i rows..", events["data"].shape[0])

        #### Objects ####
        logging.info("      creating objects..")
        
        objects["vendor"] = {
            "data": log[["vendor", "vendor_name"]].drop_duplicates(),
            "primary_key": "vendor",
            "attribute_columns": ["vendor_name"],
            "file_path": "",
        }
        logging.info("         created vendor data with a total of %i rows..", objects["vendor"]["data"].shape[0])

        objects["company"] = {
            "data": log[["company"]].drop_duplicates(),
            "primary_key": "company",
            "attribute_columns": [],
            "file_path": "",
        }
        logging.info("         created company data with a total of %i rows..", objects["company"]["data"].shape[0])

        objects["purchase_order"] = {
            "data": log[["purchase_order", "purchase_order_doc_type"]].drop_duplicates(),
            "primary_key": "purchase_order",
            "attribute_columns": ["purchase_order_doc_type"],
            "file_path": "",
        }
        logging.info("         created purchase_order data with a total of %i rows..", objects["purchase_order"]["data"].shape[0])

        objects["purchase_order_item"] = {
            "data": log[
                [
                "purchase_order_item",
                    "item",
                    "item_type",
                    "item_category",
                    "has_invoice_after_goods_receipt",
                    "has_goods_receipt",
                    "purchase_area",
                    "subpurchase_area",
                    "spend_classification",
                ]
            ].drop_duplicates(),
            "primary_key": "purchase_order_item",
            "attribute_columns": [
                "item",
                "item_type",
                "item_category",
                "has_invoice_after_goods_receipt",
                "has_goods_receipt",
                "purchase_area",
                "subpurchase_area",
                "spend_classification",
            ],
            "file_path": "",
        }
        logging.info("         created purchase_order_item data with a total of %i rows..", objects["purchase_order_item"]["data"].shape[0])

        objects["invoice"] = {
            "data": log[["invoice", "item", "item_value"]].drop_duplicates(),
            "primary_key": "invoice",
            "attribute_columns": ["item", "item_value"],
            "file_path": "",
        }
        logging.info("         created invoice data with a total of %i rows..", objects["invoice"]["data"].shape[0])

        objects["resource"] = {
            "data": log[["resource"]].drop_duplicates(),
            "primary_key": "resource",
            "attribute_columns": [],
            "file_path": "",
        }
        logging.info("         created resource data with a total of %i rows..", objects["resource"]["data"].shape[0])

        #### Event to object relations ####
        logging.info("      creating event_to_object relations..")
        
        created_by = log[
            (log["activity"] == "vendor_creates_debit_memo")
            | (log["activity"] == "vendor_creates_invoice")
        ]
        created_by = created_by[["event", "vendor"]].drop_duplicates()
        created_by["created_by"] = "created_by_" + created_by.index.astype(str)
        event_to_object["created_by"] = {
            "data": created_by,
            "primary_key": "created_by",
            "from_foreign_key": "event",
            "to_foreign_key": "vendor",
            "file_path": "",
        }
        logging.info("         created created_by data with a total of %i rows..", created_by.shape[0])
        
        modify_purchase_order = log[["event", "purchase_order"]].drop_duplicates()
        modify_purchase_order["modify_purchase_order"] = "modify_purchase_order_" + modify_purchase_order.index.astype(str)
        event_to_object["modify_purchase_order"] = {
            "data": modify_purchase_order,
            "primary_key": "modify_purchase_order",
            "from_foreign_key": "event",
            "to_foreign_key": "purchase_order",
            "file_path": "",
        }
        logging.info("         created modify_purchase_order data with a total of %i rows..", modify_purchase_order.shape[0])
        
        modify_purchase_order_item = log[["event", "purchase_order_item"]].drop_duplicates()
        modify_purchase_order_item["modify_purchase_order_item"] = "modify_purchase_order_item_" + modify_purchase_order_item.index.astype(str)
        event_to_object["modify_purchase_order_item"] = {
            "data": modify_purchase_order_item,
            "primary_key": "modify_purchase_order_item",
            "from_foreign_key": "event",
            "to_foreign_key": "purchase_order_item",
            "file_path": "",
        }
        logging.info("         created modify_purchase_order_item data with a total of %i rows..", modify_purchase_order_item.shape[0])
        
        modify_invoice = log[["event", "invoice"]].drop_duplicates()
        modify_invoice["modify_invoice"] = "modify_invoice_" + modify_invoice.index.astype(str)
        event_to_object["modify_invoice"] = {
            "data": modify_invoice,
            "primary_key": "modify_invoice",
            "from_foreign_key": "event",
            "to_foreign_key": "invoice",
            "file_path": "",
        }
        logging.info("         created modify_invoice data with a total of %i rows..", modify_invoice.shape[0])
        
        executed_by = log[["event", "resource"]].drop_duplicates()
        executed_by["executed_by"] = "executed_by_" + executed_by.index.astype(str)
        event_to_object["executed_by"] = {
            "data": executed_by,
            "primary_key": "executed_by",
            "from_foreign_key": "event",
            "to_foreign_key": "resource",
            "file_path": "",
        }
        logging.info("         created executed_by data with a total of %i rows..", executed_by.shape[0])

        #### Object to object relations ####
        logging.info("      creating object_to_object relations..")
        
        sells = log[["vendor", "purchase_order"]].drop_duplicates()
        sells["sells"] = "sells_" + sells.index.astype(str)
        object_to_object["sells"] = {
            "data": sells,
            "primary_key": "sells",
            "from_foreign_key": "vendor",
            "to_foreign_key": "purchase_order",
            "file_path": "",
        }
        logging.info("         created sells data with a total of %i rows..", sells.shape[0])
        
        buys = log[["company", "purchase_order"]].drop_duplicates()
        buys["buys"] = "buys_" + buys.index.astype(str)
        object_to_object["buys"] = {
            "data": buys,
            "primary_key": "buys",
            "from_foreign_key": "company",
            "to_foreign_key": "purchase_order",
            "file_path": "",
        }
        logging.info("         created buys data with a total of %i rows..", buys.shape[0])
        
        has_item = log[["purchase_order", "purchase_order_item"]].drop_duplicates()
        has_item["has_item"] = "has_item_" + has_item.index.astype(str)
        object_to_object["has_item"] = {
            "data": has_item,
            "primary_key": "has_item",
            "from_foreign_key": "purchase_order",
            "to_foreign_key": "purchase_order_item",
            "file_path": "",
        }
        logging.info("         created has_item data with a total of %i rows..", has_item.shape[0])
        
        invoice_for = log[log["activity"] == "Vendor_creates_invoice"]
        invoice_for = log[["purchase_order_item", "invoice"]].drop_duplicates()
        invoice_for["invoice_for"] = "invoice_for_" + invoice_for.index.astype(str)
        object_to_object["invoice_for"] = {
            "data": invoice_for,
            "primary_key": "invoice_for",
            "from_foreign_key": "invoice",
            "to_foreign_key": "purchase_order_item",
            "file_path": "",
        }
        logging.info("         created invoice_for data with a total of %i rows..", invoice_for.shape[0])
        
        data = {
            "events": events,
            "objects": objects,
            "relations": {
                "event_to_object": event_to_object,
                "object_to_object": object_to_object,
            },
        }

        return data
