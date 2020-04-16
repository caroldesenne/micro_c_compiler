class uCType(object):
    '''
    Class that represents a type in the uC language.  Types 
    are declared as singleton instances of this type.
    '''
    def __init__(self, name, bin_ops=set(), un_ops=set()):
        '''
        You must implement yourself and figure out what to store.
        '''
        self.name = name
        self.bin_ops = bin_ops
        self.un_ops = un_ops


# Create specific instances of types. You will need to add
# appropriate arguments depending on your definition of uCType
int_type = uCType("int",
    set(('PLUS', 'MINUS', 'TIMES', 'DIVIDE',
         'LE', 'LT', 'EQ', 'NE', 'GT', 'GE')),
    set(('PLUS', 'MINUS')),
    )
float_type = uCType("float",
    set(('PLUS', 'MINUS', 'TIMES', 'DIVIDE',
         'LE', 'LT', 'EQ', 'NE', 'GT', 'GE')),
    set(('PLUS', 'MINUS')),
    )
char_type = uCType("char",
    set(('PLUS',)),
    set(),
    )
string_type = uCType("string",
    set(('PLUS',)),
    set(),
    )
boolean_type = uCType("bool",
    set(('AND', 'OR', 'EQ', 'NE')),
    set(('NOT',))
    )
# In your type checking code, you will need to reference the
# above type objects.   Think of how you will want to access
# them.