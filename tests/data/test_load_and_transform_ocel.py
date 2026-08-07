import logging.config
import tempfile
import unittest
from pathlib import Path

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


_P2P_LOG = Path("data/ocel/P2P.xml")


@unittest.skipUnless(_P2P_LOG.is_file(), f"sample log {_P2P_LOG} not available")
class TestOCELPipeline(unittest.TestCase):
    """Smoke test: verifies the full OCEL→ontology pipeline runs without error."""

    def test_p2p_pipeline(self):
        settings = OntoBuildSettings()
        settings.profile = OwlProfile.OWL2EL
        settings.inverse_representation = [InverseRepresentation.IMPLICIT]
        settings.order_representation = []
        settings.case_concepts = []

        loader = PM4PYLoader(str(_P2P_LOG), {})
        oced = loader.get_oced()
        oced = clean_oced_data(oced)

        builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
        builder.build_ontology(oced)

        with tempfile.TemporaryDirectory() as tmp_dir:
            builder.save_onto(str(Path(tmp_dir) / "tmp.owl"))

        self.assertGreater(len(list(builder.onto.classes_in_signature())), 0)
        self.assertGreater(len(list(builder.onto.individuals_in_signature())), 0)


if __name__ == "__main__":
    unittest.main()
