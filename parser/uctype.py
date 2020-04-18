class uCType(object):
    '''
    Class that represents a type in the uC language.  Types 
    are declared as singleton instances of this type.
    '''
    def __init__(self, name, bin_ops=set(), un_ops=set(), bool_ops=set(), as_ops=set()):
        self.name = name
        self.bin_ops = bin_ops
        self.un_ops = un_ops
        self.bool_ops = bool_ops
        self.assign_ops = as_ops


# Create specific instances of types. You will need to add
# appropriate arguments depending on your definition of uCType
int_type = uCType(name     = "int",
                  bin_ops  = {'+', '-', '*', '/','%','==','!=','<','>','<=','>='},
                  un_ops   = {'-','+','--','++','p--','p++','*','&'},
                  bool_ops = {'==','!=','<','>','<=','>='},
                  as_ops   = {'=','+=','-=','*=','/=','%='}
                 )

float_type = uCType(name     = "float",
                    bin_ops  = {'+', '-', '*', '/','%','==','!=','<','>','<=','>='},
                    un_ops   = {'-','+','*','&'},
                    bool_ops = {'==','!=','<','>','<=','>='},
                    as_ops   = {'=','+=','-=','*=','/=','%='}
                   )

boolean_type = uCType(name     = "bool",
                      bin_ops  = {'==','!=','&&','||'},
                      un_ops   = {'!','*','&'},
                      bool_ops = {'==','!=','&&','||'},
                      as_ops   = {}
                     )

char_type = uCType(name     = "char",
                   bin_ops  = {'==','!='},
                   un_ops   = {'*','&'},
                   bool_ops = {'==','!='}, # TODO: sao so esses?
                   as_ops   = {}
                  )

string_type = uCType(name     = "string",
                     bin_ops  = {'==','!='},
                     un_ops   = {},
                     bool_ops = {'==','!='},
                     as_ops   = {}
                    )

array_type = uCType(name = "array",
                    bin_ops  = {'==','!='},
                    un_ops   = {'*','&'},
                    bool_ops = {'==','!='},
                    as_ops   = {}
                    )
# TODO pointer
pointer_type = uCType(name = "pointer",
                      bin_ops  = {},
                      un_ops   = {},
                      bool_ops = {},
                      as_ops   = {}
                     )
