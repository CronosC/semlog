import owlready2 as owl

class ReasonerInterface():
    
    def __init__(self, onto):
        with onto:
            owl.sync_reasoner_pellet()
        self.onto = onto
            
    def get_instances(self, class_name, as_str=True):
        class_entity = self.onto[class_name]
        instances = class_entity.instances()
        if as_str:
            instances = self.onto_entities_to_string(instances)
        return instances
    
    def get_superclasses(self, thing_name, as_str=True):
        thing_entity = self.onto[thing_name]
        things = thing_entity.is_a
        if as_str:
            things = self.onto_entities_to_string(things)
        return things
    
    def get_subclasses(self, thing_name, as_str=True):
        thing_entity = self.onto[thing_name]
        things = thing_entity.subclasses()
        if as_str:
            things = self.onto_entities_to_string(things)
        return things
    
    def onto_entities_to_string(self, entities, as_str=True):
        return [e.name for e in entities]