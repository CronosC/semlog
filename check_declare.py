from semanticlog.enrich.declare import DeclareEnricher
from semanticlog.data.classicalloader import ClassicalLoader
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.data.cleaning import clean_oced_data

cl = ClassicalLoader(
    "data/au_timestamp_start.csv",
    {"case_col": "video_id", "act_col": "active_au", "time_col": "timestamp_start"},
)


oced = cl.get_oced()
oced = clean_oced_data(oced)

de = DeclareEnricher(oced)
#declare_df = de.get_constraints_via_discovery(
#    ["video_id"],
#    vacuuos_is_considered=False,
#    constraint_support_threshold=0.3,
#    itemset_support_threshold=0.3,
#)
declare_df = de.get_constraints_via_enumeration(
    ["video_id"],
    vacuuos_is_considered=False,
    constraint_set=["Existence1", "Existence2", "Existence3", "Init", "End", "Response", "Precedence", "Alternate Response", "Alternate Precedence"]
)

print(declare_df)

declare_df.to_csv("data/emotions_declare.csv", index=False)
