import logging
import cython
import owlready2 as owl
from rdflib import Graph

def save_onto(onto, path, save_as="xml"):
    with onto:
        if save_as=="xml":
            logging.info("saving ontology to file %s.xml..", path)
            onto.save(file = path + ".xml", format = "rdfxml")
        elif save_as=="ttl":
            logging.info("saving ontology to file %s.ttl..", path)
            graph = owl.default_world.as_rdflib_graph()
            graph.serialize(destination=path + ".ttl", format='turtle')
        elif save_as=="both":
            save_onto(onto, path, save_as="xml")
            save_onto(onto, path, save_as="ttl")
            
def load_onto(path,  load_from="xml"):
    if load_from=="xml":
            logging.info("loading ontology from file %s.xml..", path)
            onto = owl.get_ontology(path + ".xml").load()
    else:
        raise ValueError("Unidentified format to load ontology from: ", load_from)
    return onto
