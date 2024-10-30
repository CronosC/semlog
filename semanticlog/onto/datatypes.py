import cython
import owlready2 as owl

def setup_datatypes(onto):
    """makes the necessary datatypes available to owlready"""
    with onto:
        owl.declare_datatype(NonNegInt, "http://www.w3.org/2001/XMLSchema#nonNegativeInteger", parse_non_negative_int, unparse_non_negative_int)
        owl.declare_datatype(gYear, "http://www.w3.org/2001/XMLSchema#gYear", parse_gYear, unparse_gYear)
        owl.declare_datatype(gMonth, "http://www.w3.org/2001/XMLSchema#gMonth", parse_gMonth, unparse_gMonth)
        owl.declare_datatype(gDay, "http://www.w3.org/2001/XMLSchema#gDay", parse_gDay, unparse_gDay)
    return onto

#### Datatype definitions and parse/unparse functions ####

class NonNegInt(object):
    def __init__(self, value):
        self.value = value

def unparse_non_negative_int(nn_int):
    return str(nn_int.value)

def parse_non_negative_int(string):
    return NonNegInt(abs(int(string)))

class gYear(object):
    def __init__(self, value):
        self.value = value

def unparse_gYear(gyear):
    return "{:04d}".format(int(gyear.value))

def parse_gYear(string):
    return gYear(int(string))

class gMonth(object):
    def __init__(self, value):
        self.value = value

def unparse_gMonth(gmonth):
    return "--{:02d}".format(int(gmonth.value))

def parse_gMonth(string):
    return gMonth(int(string.lstrip("--")))

class gDay(object):
    def __init__(self, value):
        self.value = value

def unparse_gDay(gday):
    return "---{:02d}".format(int(gday.value))

def parse_gDay(string):
    return gDay(int(string.lstrip("---")))