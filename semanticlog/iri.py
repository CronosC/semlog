"""The default ontology namespace, in one place.

A leaf module with no imports of its own. Both the data layer and the ontology
layer need this constant, and neither can import it from the other:
``semanticlog.onto`` pulls in the whole build stack on first import, so reaching
for it from ``semanticlog.data`` closes an import cycle.
"""

#: Namespace every local name is appended to unless a build says otherwise.
#:
#: Not a resolvable URL. Publishing these ontologies would want a real
#: host-based namespace, but changing it renames every IRI in every file already
#: produced, so it stays until someone decides to make that break.
DEFAULT_IRI: str = "https://oced#"
