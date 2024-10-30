import owlready2 as owl

class QueryInterface:
    """
    A class to interface with an ontology and perform SPARQL queries.

    Attributes:
        onto: The ontology instance on which queries will be executed.
    """

    def __init__(self, onto):
        self.onto = onto
            
    #### GENERIC ####

    def run_query(self, query_str):
        """
        Executes a SPARQL query on the ontology and returns the unique results.

        Args:
            query_str (str): The SPARQL query string to execute.
        Returns:
            list: A list of unique result elements from the query, with object names extracted if applicable.
        """
        query_str = (
            "PREFIX OCED: <https://uni-bamberg.de/KIGA/OCED.owl#> \n" + query_str
        )
        world = self.onto.world
        results = [
            (element if type(element) == str else element.name)
            for sublist in list(world.sparql(query_str))
            for element in sublist
        ]
        return list(set(results))
    
    def run_query_from_file(self, file_path):
        with open(file_path, 'r') as file:
            query = file.read()
        return self.run_query(query)

    #### SPECIALIZED ####

    def get_traces(self):
        """
        Retrieves all traces from the ontology by running the corresponding SPARQL query.

        Returns:
            list: A list of unique trace identifiers.
        """
        return self.run_query(self.prepare_get_all_traces_query)

    def get_attribute_values(self, object_attribute_type):
        """
        Retrieves all values for a specified object attribute type.

        Args:
            object_attribute_type (str): The class of the object attribute to retrieve values for.
        Returns:
            list: A list of unique attribute values.
        """
        return self.run_query(
            self.prepare_get_all_attribute_values_query(object_attribute_type)
        )

    def get_traces_via_object_attribute(
        self, object_attribute_type, object_attribute_value
    ):
        """
        Retrieves traces linked to objects with a specific attribute type and value.

        Args:
            object_attribute_type (str): The attribute class to filter by.
            object_attribute_value (str): The specific value of the attribute to filter by.
        Returns:
            list: A list of unique trace identifiers matching the criteria.
        """
        return self.run_query(
            self.prepare_get_traces_by_object_attribute_value_query(
                object_attribute_type, object_attribute_value
            )
        )
        
    def get_declare_constraints(self):
        '''
        Retrieves all unique Declare Constraint classes from the ontology.
        
        Returns:
            list: A list of unique Declare Constraint classes.
        '''
        
        query_str = ''' 
            SELECT ?c
            WHERE {
                ?c rdfs:subClassOf OCED:DeclareConstraint
            }
        '''
        return self.run_query(query_str)
    
    def get_individuals_by_class(self, class_str):
        '''
        Retrieves all unique individuals of a given class.
        
        Args:
            class_str (str): The string name of the class for which individuals are requested.
        Returns:
            list: A list of all unique individuals belonging to some class.
        '''
        query_str = ''' 
            SELECT ?i
            WHERE {
                ?i a OCED:<<Class>>
            }
        '''.replace("<<Class>>", class_str)
        return self.run_query(query_str)
    
    def get_subclasses(self, superclass_str):
        '''
        Retrieves all subclasses of a given class.
        
        Args:
            class_str (str): The string name of the class for which subclasses are requested.
        Returns:
            list: A list of all unique subclasses belonging to some class.
        '''
        query_str = ''' 
            SELECT ?i
            WHERE {
                ?i rdfs:subClassOf OCED:<<Class>>
            }
        '''.replace("<<Class>>", superclass_str)
        return self.run_query(query_str)
        
        
    #### QUERY PREP ####

    def prepare_get_all_traces_query(self):
        """
        Prepares a SPARQL query string to retrieve all traces.

        Returns:
            str: The SPARQL query string that retrieves all trace individuums.
        """

        query_str = """
            SELECT ?id
            WHERE {
                ?id a OCED:Trace
            }
        """
        return query_str

    def prepare_get_traces_by_object_attribute_value_query(
        self, object_attribute_type, object_attribute_value
    ):
        """
        Prepares a SPARQL query string to retrieve traces based on an object's attribute type and value.

        Args:
            object_attribute_type (str): The class of the object attribute being searched (e.g., "OCED:Color").
            object_attribute_value (str): The specific value of the object attribute being searched (e.g., "Blue").
        Returns:
            str: The SPARQL query string that retrieves traces based on the object attribute type and value.
        """

        query_str = """ 
            SELECT ?T
            WHERE {
                ?T a OCED:Trace .
                ?R rdfs:subPropertyOf OCED:hasAttribute .
                ?T ?R ?A .
                ?A a OCED:<<object_attribute_type>> .
                ?A OCED:hasValue "<<object_attribute_value>>"
            }
        """

        query_str = query_str.replace(
            "<<object_attribute_type>>", object_attribute_type
        )
        query_str = query_str.replace(
            "<<object_attribute_value>>", object_attribute_value
        )
        return query_str

    def prepare_get_all_attribute_values_query(self, object_attribute_type):
        """
        Prepares a SPARQL query string to retrieve all values of a specific object attribute type.

        Args:
            object_attribute_type (str): The class of the object attribute whose values are being retrieved.
        Returns:
            str: The SPARQL query string that retrieves all values of the specified object attribute type.
        """

        query_str = """ 
            SELECT ?V
            WHERE {
                ?A a OCED:<<object_attribute_type>> .
                ?A OCED:hasValue ?V
            }
        """

        query_str = query_str.replace(
            "<<object_attribute_type>>", object_attribute_type
        )
        return query_str
