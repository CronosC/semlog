class DeclareShorthands:
    """
    This class provides a set of predefined shorthand notations for Declare constraints.
    
    The shorthands are mappings of shorthand names (like "n1", "reex", etc.) to sets of Declare constraints. 
    It offers methods to retrieve the shorthand's corresponding set of constraints and to split constraints into 
    unary and binary categories.
    
    Attributes:
        shorthands (dict): A dictionary mapping shorthand names to lists of Declare constraints.
    """
    # for the commented shorthands model checking doesn't work? seems to be because they are defined in terms
    # of other constraints?
    shorthands = {
        "n1": ["Absence1", "Existence1"],
        "n2": ["Absence1", "Existence1", "Absence2", "Existence2"],
        "n3": [
            "Absence1",
            "Existence1",
            "Absence2",
            "Existence2",
            "Absence3",
            "Existence3",
        ],
        "n1+": ["Absence1", "Existence1", "Exactly1"],
        "n2+": [
            "Absence1",
            "Existence1",
            "Exactly1",
            "Absence2",
            "Existence2",
            "Exactly2",
        ],
        "n3+": [
            "Absence1",
            "Existence1",
            "Exactly1",
            "Absence2",
            "Existence2",
            "Exactly2",
            "Absence3",
            "Existence3",
            "Exactly3",
        ],
        "se": ["End", "Init"],
        "reex": ["Responded Existence"],
        "nreex": ["Responded Existence", "Not Responded Existence"],
        # "coex": ["Co-Existence"],
        # "ncoex": ["Co-Existence", "Not Co-Existence"],
        "re": ["Response"],
        "nre": ["Response", "Not Response"],
        "reca": ["Response", "Alternate Response"],
        "recc": ["Response", "Chain Response"],
        "rec": ["Response", "Alternate Response", "Chain Response"],
        "nrecc": ["Response", "Chain Response", "Not Response", "Not Chain Response"],
        "pre": ["Precedence"],
        "npre": ["Precedence", "Not Precedence"],
        "preca": ["Precedence", "Alternate Precedence"],
        "precc": ["Precedence", "Chain Precedence"],
        "prec": ["Precedence", "Alternate Precedence", "Chain Precedence"],
        "nprecc": [
            "Precedence",
            "Chain Precedence",
            "Not Precedence",
            "Not Chain Precedence",
        ],
        # "su": ["Succession"],
        # "nsu": ["Succession", "Not Succession"],
        # "suc": ["Succession", "Chain Succession"],
        "or": ["Choice"],
        "xor": ["Exclusive Choice"],
    }

    def __init__(self):
        pass
    
    def is_shorthand(self, declare_string):
        return declare_string in self.shorthands.keys()

    def get(self, shorthand, split=False):
        """
        Returns the constraints for the given shorthand. If split=True, splits the constraints 
        into unary and binary sets.
        
        Args:
            shorthand (str): The shorthand to look up.
            split (bool): If True, splits constraints into unary and binary.
            
        Returns:
            list or tuple: A list of constraints or a tuple of two lists (unary, binary).
        """
        if split:
            return self.split_unary_binary(self.shorthands[shorthand])
        else:
            return self.shorthands[shorthand]
        
    def get_all(self, shorthands, split=False):
        unfolded_shorthands = []
        for constraint in shorthands:
            if self.is_shorthand(constraint):
                unfolded_shorthands += self.get(constraint)
            else:
                unfolded_shorthands += constraint
      
        if split:
            return self.split_unary_binary(unfolded_shorthands)
        else:
            return unfolded_shorthands

    def split_unary_binary(self, declare_set):
        """
        Splits the given constraint set into unary and binary constraints.
        
        Args:
            declare_set (list): The set of constraints to split.
            
        Returns:
            tuple: A tuple containing two lists (unary constraints, binary constraints).
        """
        unary_constraints = [
            "Absence1",
            "Existence1",
            "Exactly1",
            "Absence2",
            "Existence2",
            "Exactly2",
            "Absence3",
            "Existence3",
            "Exactly3",
            "End",
            "Init",
        ]
        split_constraint_sets = [[], []]
        for constraint in declare_set:
            if constraint in unary_constraints:
                split_constraint_sets[0] += [constraint]
            else:
                split_constraint_sets[1] += [constraint]
        return split_constraint_sets
