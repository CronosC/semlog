import logging.config
import os
import unittest

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.loading.pm4pyloader import PM4PYLoader
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    InverseRepresentation,
    OntoBuildSettings,
    OwlProfile,
)

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)
logging.root.handlers = []
logging.basicConfig(
    format="%(name)s | %(asctime)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)


class TestOCELPipeline(unittest.TestCase):
    """Smoke test: verifies the full OCEL→ontology pipeline runs without error."""

    def test_p2p_pipeline(self):
        out_path = "data/temp/tmp.owl"

        settings = OntoBuildSettings()
        settings.profile = OwlProfile.OWL2EL
        settings.inverse_representation = [InverseRepresentation.IMPLICIT]
        settings.order_representation = []
        settings.case_concepts = []

        loader = PM4PYLoader("data/ocel/P2P.xml", {})
        oced = loader.get_oced()
        oced = clean_oced_data(oced)

        builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
        builder.build_ontology(oced)
        builder.save_onto(out_path)

        self.assertGreater(len(list(builder.onto.classes_in_signature())), 0)
        self.assertGreater(len(list(builder.onto.individuals_in_signature())), 0)

        os.remove(out_path)


if __name__ == "__main__":
    unittest.main()
