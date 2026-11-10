"""Turning stored event data into an OWL 2 ontology.

`settings.OntoBuildSettings` says what to emit, `builder.OCEDOntoBuilder`
emits it through a sink, and `pipeline.build_log` runs one build from a
cleaned store to a file. `reason.Reasoner` starts the Java OWL API.
"""
