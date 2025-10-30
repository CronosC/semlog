from semanticlog.onto.reason import Reasoner
from owlapy import owl_expression_to_manchester

path = "data/ONTO/Hospital_Quantized_OWL2DL_DF.owl"

out_path = path.split(".")[0] + "_MAT.owl"

r = Reasoner(path)
r.materialize(overwrite_old=False)
r.save_onto(save_path=out_path)