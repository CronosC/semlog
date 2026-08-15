# Changelog

The version is annotated onto every ontology as `owl:versionInfo`, so an existing file says which release built it.

## 0.6.0

An ontology built with 0.6 is not byte-identical to the one 0.5 built from the same log and config. Every change is to the shape of the file, and nothing an existing config sets has to change.

- The build record is written inside the `Ontology(...)` header, as annotations on the ontology itself. 0.5 wrote it as annotation assertions naming the ontology IRI, which OWL functional syntax does not read back as ontology metadata. Reading the record through the OWL API or rdflib is unchanged, and asking for the ontology annotations now finds it.
- The annotation properties of that record are declared. Without the declarations a streamed build was outside OWL 2 DL, whatever `profile` said.
- An `OWL2EL` build writes a float as `xsd:decimal`, and a boolean and a date as `xsd:string`. Those three types are not in the OWL 2 EL datatype map, so a build carrying them was outside the profile it named.
- A number too large or too small for plain notation is written in full, in both the value individual's name and its literal. A value like `1e+20` used to fail the build, since `+` cannot appear in an IRI local name.
- An infinite or missing number is refused with a message naming the attribute, rather than becoming an individual called `amount_inf`.
- `iri` is a new setting. It defaults to what every previous build used, so an existing config produces the same namespace it always did.

## 0.5.0

Two `order_representations` settings changed and the output file is named differently, so an ontology built with 0.5 is not always the one 0.4 built from the same log and config.

- A streamed build writes `<name>.ofn` where 0.4 wrote `<name>.owl`. The contents are the same functional syntax. `--no-streaming` still writes `<name>.owl`, and it is still RDF/XML.
- `ef` on its own is refused. It expresses eventually-follows as a transitivity axiom over directly-follows, so with no `df` beside it a reasoner has nothing to derive the pairs from. Write `["df", "ef"]`.
- `ef_explicit` on its own no longer emits directly-follows. It asserts every pair itself and needs nothing beside it. Write `["df", "ef_explicit"]` to keep what 0.4 produced.
- The TOML keys were renamed: `order` to `order_representations`, `size_limit` to `size_limit_mb`, and the two other representation settings to their plural forms. An old key is reported as unknown, with the new name suggested.
