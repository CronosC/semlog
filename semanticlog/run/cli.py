import click
from semanticlog.data.pipe import DataPipeline
from semanticlog.data.unflatten import Unflattener_BPIC19
from semanticlog.onto.upper import OCEDUpperOntologyBuilder
from semanticlog.onto.instantiate import OCEDInstantiator
from semanticlog.onto.misc import save_onto
from semanticlog.enrich.declare import DeclareEnricher
import logging
import warnings
import pm4py as pm
import pandas as pd
import os


@click.argument("name", required=True)
@click.option(
    "--out",
    default="../../data/logs/",
    help="directory under which data should be found/saved, the standart is /data/logs/",
)
@click.group()
@click.pass_context
def main(context, name, out):
    """
    Main entrypoint for the semantic log functionality. Takes a name, which identifies the data, optionally a folder,
    where it will be saved, and finally some operation to do, which might require additional arguments.\n\n
    :param str name: The name under which the data created or used will be identified.\n
    """
    logging.root.handlers = []
    logging.basicConfig(
        format="%(asctime)s | %(message)s", level=logging.INFO, datefmt="%H:%M:%S"
    )
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, out, name)
    context.ensure_object(dict)
    context.obj["name"] = name
    context.obj["out_path"] = out_path


#### Preparation steps ####


@main.command()
@click.option(
    "--months",
    default=10,
    help="the number of months to keep in the data, must be between 1 and 10",
)
@click.option(
    "--max_variant_repetitions",
    default=10,
    help="the allowed maximum number of different traces a variant may have",
)
@click.option(
    "--min_traces_per_variant",
    default=2,
    help="the allowed minimum number of traces that a variant must have",
)
@click.pass_context
def prep_bpic19(context, months, max_variant_repetitions, min_traces_per_variant):
    """prepares the BPIC19 challenge log xes data"""
    min_month = 7 - int(months / 2)
    max_month = 6 + int(months / 2)
    months_to_remove = list(range(1, min_month)) + list(range(max_month + 1, 13))

    pipe = DataPipeline(context.obj["out_path"], load=False)
    unflattener = Unflattener_BPIC19(
        "data/xes/BPI_Challenge_2019.xes",
        months_to_remove=months_to_remove,
        max_variant_repetitions=max_variant_repetitions,
        min_traces_per_variant=min_traces_per_variant,
    )
    pipe.data = unflattener.data
    pipe.save()

    pm.write_xes(
        unflattener.log,
        os.path.join(context.obj["out_path"], context.obj["name"] + ".xes"),
        case_id_key="purchase_order",
    )


@main.command()
def add_declare():
    """enriching with DECLARE constraints is not yet implemented"""
    click.echo("Enriching with DECLARE constraints is not yet implemented.")


@main.command()
def add_sle():
    """enriching with system-level events is not yet implemented"""
    click.echo("Enriching with system-level events is not yet implemented.")


#### Ontology instantiation steps ####


@main.command()
@click.pass_context
def build_onto(context):
    """rebuilds the OCED upper ontology and then instantiates it with data"""
    onto_upper = OCEDUpperOntologyBuilder().onto
    upper_path = os.path.join(context.obj["out_path"], "OCED-upper")
    save_onto(onto_upper, upper_path)

    pipe = DataPipeline(context.obj["out_path"])

    instantiator = OCEDInstantiator()
    onto = instantiator.instantiate(onto_upper, pipe.data, context.obj["name"])

    onto_path = os.path.join(context.obj["out_path"], "OCED-" + context.obj["name"])
    save_onto(onto, onto_path)


#### Enriching ####

@main.command()
@click.pass_context
def enrich_declare(context):
    """adds declare constraint data for each of the case_concepts"""
    pipe = DataPipeline(folderpath=context.obj["out_path"], ignore_declare=True)
    pipe.add_all_declare_constraints((["Existence"], ["Responded Existence"]))
