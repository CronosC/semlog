import click
from semanticlog.data.pipe import DataPipeline
from semanticlog.data.unflatten import Unflattener_BPIC19
from semanticlog.data.unflatten import Unflattener_XES
from semanticlog.data.unflatten import XES_Preprocessor
from semanticlog.onto.upper import OCEDUpperOntologyBuilder
from semanticlog.onto.instantiate import OCEDInstantiator
from semanticlog.onto.misc import save_onto
from semanticlog.onto.misc import load_onto
from semanticlog.enrich.declare import DeclareEnricher
from semanticlog.learn.dllearner import DLLearnerPipe
from semanticlog.enrich.embed import EmbeddingEnricher
from semanticlog.onto.query import QueryInterface
from semanticlog.onto.reason import ReasonerInterface
from semanticlog.data.shorthands import DeclareShorthands
from semanticlog.data.misc import get_weighted_variant_mapping
import logging
import warnings
import pm4py as pm
import pandas as pd
import os
import json
import sys

sys.setrecursionlimit(5000)

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


#### Preparation ####


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
        "data/xes/BPIC19.xes",
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
@click.argument("xes_path", required=True)
@click.option(
    "--max_variant_repetitions",
    default=999999,
    help="the allowed maximum number of different traces a variant may have",
)
@click.option(
    "--min_traces_per_variant",
    default=1,
    help="the allowed minimum number of traces that a variant must have",
)
@click.option(
    "--additional_attribute_path",
    default=None,
    help="path to an additional csv file with a trace level attribute to include in the data",
)
@click.option(
    "--case_key",
    default="case:concept:name",
    help="column with case id information",
)
@click.option(
    "--act_key",
    default="concept:name",
    help="column with activity type information",
)
@click.option(
    "--time_key",
    default="time:timestamp",
    help="column with event timestamp information",
)
@click.pass_context
def prep_xes(context, xes_path, max_variant_repetitions, min_traces_per_variant, additional_attribute_path, case_key, act_key, time_key):
    """prepares any generic xes log, however without any knowledge about objects etc."""

    pipe = DataPipeline(context.obj["out_path"], load=False)
    unflattener = Unflattener_XES(
        xes_path,
        max_variant_repetitions=max_variant_repetitions,
        min_traces_per_variant=min_traces_per_variant,
        additional_attribute_path=additional_attribute_path,
        case_column=case_key,
        event_type_column=act_key,
        time_column=time_key
    )
    pipe.data = unflattener.data
    pipe.save()

    pm.write_xes(
        unflattener.log,
        os.path.join(context.obj["out_path"], context.obj["name"] + ".xes"),
        case_id_key="case:concept:name",
    )

@main.command()
@click.argument("log_path", required=True)
@click.pass_context
def get_variant_weights(context, log_path):
    """writes a mapping of variants to number of occurences"""
    variants = get_weighted_variant_mapping(log_path)
    
    variant_weights = {}
    pre_pro = XES_Preprocessor()
    for variant, traces in variants.items():
        variant_weights[str([pre_pro.clean_string(x) for x in variant])] = len(traces)
        
    mapping_path = os.path.join(context.obj["out_path"], "variant_weights.json")
    with open(mapping_path, "w+") as mapping_file:
        json.dump(variant_weights, mapping_file)
    

#### Ontology ####

@main.command()
@click.option(
    "--save_as",
    default="both",
    help="what file type to save the onto as, either ttl, xml or both",
)
@click.option(
    "--reason",
    default=False,
    help="whether a reasoner should be called to materialize the onto",
)
@click.pass_context
def build_onto(context, save_as, reason):
    """rebuilds the OCED upper ontology and then instantiates it with data"""
    onto_upper = OCEDUpperOntologyBuilder().onto
    upper_path = os.path.join(context.obj["out_path"], "OCED-upper")
    save_onto(onto_upper, upper_path)

    pipe = DataPipeline(context.obj["out_path"])

    instantiator = OCEDInstantiator()
    onto = instantiator.instantiate(onto_upper, pipe.data, context.obj["name"], reason=reason)

    onto_path = os.path.join(context.obj["out_path"], "OCED-" + context.obj["name"])
    save_onto(onto, onto_path, save_as=save_as)
    
    
@main.command()
@click.argument("file_path", required=True)
@click.pass_context
def onto_query(context, file_path):
    """runs sparql query read from the given file against the ontology"""
    onto_path = os.path.join(context.obj["out_path"], "OCED-" + context.obj["name"])
    onto = load_onto(onto_path)
    querier = QueryInterface(onto)
    res = querier.run_query_from_file(file_path)
    print(res)

@main.command()
@click.argument("class_name", required=True)
@click.option(
    "--get_type",
    default="ind",
    help="whether to get individiums (ind), superclasses (sup) or subclasses (sub)"
)
@click.pass_context
def onto_get(context, class_name, get_type):
    """returns individuals of a given class"""
    onto_path = os.path.join(context.obj["out_path"], "OCED-" + context.obj["name"])
    onto = load_onto(onto_path)
    reasoner = ReasonerInterface(onto)
    if get_type == "ind":
        res = reasoner.get_instances(class_name)
    elif get_type == "sup":
        res = reasoner.get_parents(class_name)
    elif get_type == "sub":
        res = reasoner.get_children(class_name)
    print(res)

#### Enriching ####

@main.command()
@click.option(
    "--vacuous",
    default=False,
    help="whether vacuously fullfilled constraints should be considered True"
)
@click.option(
    "--constraint_set", "-c",
    default=["n1"],
    multiple=True,
    help="The the set of declare constraints to mine."
)
@click.pass_context
def enrich_declare(context, vacuous, constraint_set):
    """adds declare constraint data for each of the case_concepts"""
    constraint_set = [constraint for element in constraint_set for constraint in element.split(',')]
    
    declare_lookup = DeclareShorthands()
    split_constraint_sets = declare_lookup.get_all(constraint_set, split=True)

    pipe = DataPipeline(folderpath=context.obj["out_path"], ignore_declare=True)
    pipe.add_all_declare_constraints(split_constraint_sets, consider_vacuity=vacuous)
    
@main.command()
@click.argument("object_type", required=True)
@click.argument("vectors_path", required=True)
@click.option(
    "--only_vis",
    default=False,
    help="if True the result is not added to the ontology, but just visualized via t-SNE",
)
@click.pass_context
def enrich_silent(context, object_type, vectors_path, only_vis):
    """adds silent object types for a given object, based on vector embedding clusters"""
    onto_path = os.path.join(context.obj["out_path"], "OCED-" + context.obj["name"])
    onto = load_onto(onto_path)
    
    enricher = EmbeddingEnricher(onto, vectors_path)
    silent_objects = enricher.find_silent_objects(object_type, visualize=only_vis)
    if (silent_objects is not None) and not only_vis:
        enricher.instantiate_silent_objects(silent_objects, object_type)
        
        onto_path = os.path.join(context.obj["out_path"], "OCED-" + context.obj["name"])
        save_onto(enricher.onto, onto_path, save_as="both")



@main.command()
@click.pass_context
def enrich_sle():
    """enriching with system-level events is not yet implemented"""
    click.echo("Enriching with system-level events is not yet implemented.")

    
#### Learning ####

@main.command()
@click.argument("attribute", required=True)
@click.option(
    "--time",
    default=60,
    help="the number of seconds the DLLearner searches for a theory",
)
@click.option(
    "--theory",
    default="DECLARE",
    help="what theory is allowed to construct explanations from, must be either 'DL' or 'DECLARE'."
)
@click.pass_context
def learn_attribute_explanation(context, attribute, time, theory):
    """constructs a learning problem for a given object attribute, then runs the dllearner on that learning problem"""
    
    onto_path = os.path.join(context.obj["out_path"], "OCED-" + context.obj["name"])
    onto = load_onto(onto_path)
    
    learner = DLLearnerPipe(context.obj["out_path"], onto)
    config={
        "maxExecutionTimeInSeconds": time,
    }
    results = learner.learn_attribute_explanations(attribute, concept_set=theory, alg_config=config)
    
    theory_path = os.path.join(context.obj["out_path"], "explanations_attribute.json")
    with open(theory_path, 'w+') as file:
        json.dump(results, file, indent=4)


@main.command()
@click.argument("superclass", required=True)
@click.option(
    "--time",
    default=60,
    help="the number of seconds the DLLearner searches for a theory",
)
@click.pass_context
def learn_disjoint_subclasses(context, superclass, time, only_silent=True):
    """constructs a learning problem for all subclasses of a given class and runs the dllearner on that learning problems"""
    
    onto_path = os.path.join(context.obj["out_path"], "OCED-" + context.obj["name"])
    onto = load_onto(onto_path)
    
    learner = DLLearnerPipe(context.obj["out_path"], onto)
    config={
        "maxExecutionTimeInSeconds": time,
    }
    results = learner.learn_disjoint_subclass_explanations(superclass, concept_set="DL", alg_config=config, only_silent=only_silent)
    
    theory_path = os.path.join(context.obj["out_path"], "explanations_disjoint.json")
    with open(theory_path, 'w+') as file:
        json.dump(results, file, indent=4)
    
    
    
    