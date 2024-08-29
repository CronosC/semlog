import logging
import cython
import owlready2 as owl
from rdflib import Graph

def save_onto(onto, path, save_as="ttl"):
    with onto:
        if save_as=="xml":
            logging.info("saving ontology to file(s) %s.xml..", path)
            onto.save(file = path + ".xml", format = "rdfxml")
        elif save_as=="ttl":
            logging.info("saving ontology to file(s) %s.ttl..", path)
            graph = owl.default_world.as_rdflib_graph()
            graph.serialize(destination=path + ".ttl", format='turtle')
