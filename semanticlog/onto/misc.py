import logging
import cython
import owlready2 as owl
from rdflib import Graph

logging.basicConfig(format='%(asctime)s | %(message)s', level=logging.INFO)

def save_onto(onto, path):
    logging.info("Saving ontology to file(s) %s..", path)
    with onto:
        logging.info("   saving as .xml..")
        onto.save(file = path + ".xml", format = "rdfxml")
        logging.info("   saving as .ttl..")
        graph = owl.default_world.as_rdflib_graph()
        graph.serialize(destination=path + ".ttl", format='turtle')
