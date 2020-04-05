# uc_ast
import sys

### PRE DEFINED STUFF FROM PROFESSOR MATERIAL ###

def _repr(obj):
    """
    Get the representation of an object, with dedicated pprint-like format for lists.
    """
    if isinstance(obj, list):
        return '[' + (',\n '.join((_repr(e).replace('\n', '\n ') for e in obj))) + '\n]'
    else:
        return repr(obj)

class Node(object):
    """
    Base class example for the AST nodes.

    By default, instances of classes have a dictionary for attribute storage.
    This wastes space for objects having very few instance variables.
    The space consumption can become acute when creating large numbers of instances.

    The default can be overridden by defining __slots__ in a class definition.
    The __slots__ declaration takes a sequence of instance variables and reserves
    just enough space in each instance to hold a value for each variable.
    Space is saved because __dict__ is not created for each instance.
    """
    __slots__ = ()

    def children(self):
        """ A sequence of all children that are Nodes. """
        pass

    def __repr__(self):
        """ Generates a python representation of the current node
        """
        result = self.__class__.__name__ + '('
        indent = ''
        separator = ''
        for name in self.__slots__[:-2]:
            result += separator
            result += indent
            result += name + '=' + (_repr(getattr(self, name)).replace('\n', '\n  ' + (' ' * (len(name) + len(self.__class__.__name__)))))
            separator = ','
            indent = ' ' * len(self.__class__.__name__)
        result += indent + ')'
        return result

    def show(self, buf=sys.stdout, offset=0, attrnames=False, nodenames=False, showcoord=False, _my_node_name=None):
        """
        Pretty print the Node and all its attributes and children (recursively) to a buffer.
        buf:
            Open IO buffer into which the Node is printed.
        offset:
            Initial offset (amount of leading spaces)
        attrnames:
            True if you want to see the attribute names in name=value pairs. False to only see the values.
        nodenames:
            True if you want to see the actual node names within their parents.
        showcoord:
            Do you want the coordinates of each Node to be displayed.
        """
        lead = ' ' * offset
        if nodenames and _my_node_name is not None:
            buf.write(lead + self.__class__.__name__+ ' <' + _my_node_name + '>: ')
        else:
            buf.write(lead + self.__class__.__name__+ ': ')

        if self.attr_names:
            if attrnames:
                nvlist = [(n, getattr(self, n)) for n in self.attr_names if getattr(self, n) is not None]
                attrstr = ', '.join('%s=%s' % nv for nv in nvlist)
            else:
                vlist = [getattr(self, n) for n in self.attr_names]
                attrstr = ', '.join('%s' % v for v in vlist)
            buf.write(attrstr)

        if showcoord:
            if self.coord:
                buf.write('%s' % self.coord)
        buf.write('\n')

        for (child_name, child) in self.children():
            child.show(buf, offset + 4, attrnames, nodenames, showcoord, child_name)

'''
TODO:
FuncDecl ( ) ?? there is FuncDef as well **
VarDecl () **
'''

'''
DONE:
Program ( )
GlobalDecl ( )
DeclList ( )
ArrayDecl ( )
ArrayRef ( )
ID (name)
Assignment (op)
PtrDecl ( )
FuncCall ( )
FuncDef ( ) **
BinaryOp (op)
UnaryOp (op) *
Constant (type, value)
Type (names)
Decl (name) *
InitList ( )
ExprList ( )
Compound ( )
If ( )
While ( )
For ( )
Break ( )
Return ( )
Assert ( )
Print ( )
Read ( )
EmptyStatement ( )
'''

class Program(Node):
    __slots__ = ('gdecls', 'coord')

    def __init__(self, gdecls, coord=None):
        self.gdecls = gdecls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.gdecls or []):
            nodelist.append(("gdecls[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()

class GlobalDecl(Node):
    __slots__ = ('decl','coord')

    def __init__(self, decl, coord=None):
        self.decl = decl
        self.coord = coord

    def children(self):
        nodelist = []
        nodelist.append(("decl", self.decl))
        return tuple(nodelist)

    attr_names = ()

class DeclList(Node):
    __slots__ = ('decls','coord')

    def __init__(self, decls, coord=None):
        self.decls = decls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()

class FuncDef(Node):
    __slots__ = ('type_spec','declarator','decl_list','compound_statement','coord')

    def __init__(self, ts, d, dl, cs, coord=None):
        self.type_spec = ts
        self.declarator = d
        self.decl_list = dl
        self.compound_statement = cs
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type_spec is not None: nodelist.append(("type_spec", self.type_spec))
        for i, child in enumerate(self.decl_list or []):
            nodelist.append(("declaration[%d]" % i, child))
        nodelist.append(("compound_statement", self.compound_statement))
        return tuple(nodelist)

    attr_names = ()

class ParamList(Node):
    __slots__ = ('list')

    def __init__(self,l,coord=None):
        self.list = l
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.list or []):
            nodelist.append(("parameter[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()

class ArrayDecl(Node):
    __slots__ = ('declarator','size','coord')

    def __init__(self, dec, s, coord=None):
        self.declarator = dec
        self.size = s
        self.coord = coord

    def children(self):
        nodelist = []
        if self.declarator is not None: nodelist.append(("declarator", self.declarator))
        if self.size is not None: nodelist.append(("size", self.size))
        return tuple(nodelist)

    attr_names = ()

class ArrayRef(Node):
    __slots__ = ('name','access_value','coord')

    def __init__(self, name, av, coord=None):
        self.name = name
        self.access_value = av
        self.coord = coord

    def children(self):
        nodelist = []
        if self.name is not None: nodelist.append(("name", self.name))
        if self.access_value is not None: nodelist.append(("access_value", self.access_value))
        return tuple(nodelist)

    attr_names = ()

class FuncCall(Node):
    __slots__ = ('name','params','coord')

    def __init__(self, name, params, coord=None):
        self.name = name
        self.params = params
        self.coord = coord

    def children(self):
        nodelist = []
        if self.name is not None: nodelist.append(("name", self.name))
        if self.params is not None: nodelist.append(("params", self.params))
        return tuple(nodelist)

    attr_names = ()


class ID(Node):
    __slots__ = ('name','coord')

    def __init__(self,name,coord=None):
        self.name = name

    def children(self):
        nodelist = []
        return tuple(nodelist)

    attr_names = ('name', )

class Assignment(Node):
    __slots__ = ('op','assignee','value','coord')

    def __init__(self,op,ass,v,coord=None):
        self.op = op
        self.assignee = ass
        self.value = v
        self.coord = coord

    def children(self):
        nodelist = []
        if self.assignee is not None: nodelist.append(("assignee", self.assignee))
        if self.value is not None: nodelist.append(("value", self.value))
        return tuple(nodelist)

    attr_names = ('op', )

class PtrDecl(Node):
    __slots__ = ('stars','coord')

    def __init__(self, stars, coord=None):
        self.stars = stars
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    attr_names = ()

class BinaryOp(Node):
    __slots__ = ('op', 'lvalue', 'rvalue', 'coord')

    def __init__(self, op, left, right, coord=None):
        self.op = op
        self.lvalue = left
        self.rvalue = right
        self.coord = coord

    def children(self):
        nodelist = []
        if self.lvalue is not None: nodelist.append(("lvalue", self.lvalue))
        if self.rvalue is not None: nodelist.append(("rvalue", self.rvalue))
        return tuple(nodelist)

    attr_names = ('op', )

class Cast(Node):
    __slots__ = ('type','expression','coord')

    def __init__(self, t, exp, coord=None):
        self.type = t
        self.expression = exp
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        if self.expression is not None: nodelist.append(("expression", self.expression))
        return tuple(nodelist)

    attr_names = ()

class UnaryOp(Node):
    # TODO: fazer com PLUSPLUS E MINUSMINUS?? nao esta claro!!
    #__slots__ = ('op',???)
    attr_names = ('op', )

class Constant(Node):
    __slots__ = ('type', 'value', 'coord')

    def __init__(self, type, value, coord=None):
        self.type = type
        self.value = value
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    attr_names = ('type', 'value', )

class Type(Node):
    __slots__ = ('names','coord')

    def __init__(self, names, coord=None):
        self.names = names
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    attr_names = ('names', )

class VarDecl(Node):
    __slots__ = ('name','type','coord')

    def __init__(self, t, coord=None):
        self.name = None
        self.type = t
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    attr_names = ('name', )

class Decl(Node):
    __slots__ = ('name', 'type', 'init_dec_l', 'coord')

    def __init__(self, t, idl, coord=None):
        self.name = None
        self.type = t
        self.init_dec_l = idl
        self.coord = coord
        print('--', idl)

    def children(self):
        nodelist = []
        # nodelist.append(self.init_dec_l[0][0])
        # nodelist.append(self.init_dec_l[0][1])
        return tuple(nodelist)

    attr_names = ('name', )

class InitList(Node):
    __slots__ = ('inits','coord')

    def __init__(self, inits, coord=None):
        self.inits = inits
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.inits or []):
            nodelist.append(("initializer[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()

class Compound(Node):
    __slots__ = ('decl_list','st_list','coord')

    def __init__(self, dl, sl, coord=None):
        self.decl_list = dl
        self.st_list = sl
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decl_list or []):
            nodelist.append(("decl[%d]" % i, child))
        for i, child in enumerate(self.st_list or []):
            nodelist.append(("statement[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()

class If(Node):
    __slots__ = ('cond','statement','else_st','coord')

    def __init__(self, cond, st, else_st=None, coord=None):
        self.cond = cond
        self.statement = st
        self.else_st = else_st
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None: nodelist.append(("condition", self.cond))
        if self.statement is not None: nodelist.append(("statement", self.statement))
        if self.else_st is not None: nodelist.append(("else_statment", self.else_st))
        return tuple(nodelist)

    attr_names = ()

class While(Node):
    __slots__ = ('cond','statement','coord')

    def __init__(self, cond, st, coord=None):
        self.cond = cond
        self.statement = st
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None: nodelist.append(("condition", self.cond))
        if self.statement is not None: nodelist.append(("statement", self.statement))
        return tuple(nodelist)

    attr_names = ()

class For(Node):
    __slots__ = ('init','stop_cond','increment','statement','coord')

    def __init__(self, init, sc, inc, st, coord=None):
        self.init = init
        self.stop_cond = sc
        self.increment = inc
        self.statement = st
        self.coord = coord

    def children(self):
        nodelist = []
        if self.init is not None: nodelist.append(("init", self.init))
        if self.stop_cond is not None: nodelist.append(("stop_cond", self.stop_cond))
        if self.increment is not None: nodelist.append(("increment", self.increment))
        if self.statement is not None: nodelist.append(("statement", self.statement))
        return tuple(nodelist)

    attr_names = ()

class ExprList(Node):
    __slots__ = ('list','coord')

    def __init__(self,l,coord=None):
        self.list = l
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.list or []):
            nodelist.append(("expression[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()

class Break(Node):
    __slots__ = ('coord')

    def __init__(self,coord=None):
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    attr_names = ()

class Return(Node):
    __slots__ = ('expr','coord')

    def __init__(self,expr=None,coord=None):
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None: nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    attr_names = ()

class Assert(Node):
    __slots__ = ('expr','coord')

    def __init__(self,expr,coord=None):
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None: nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    attr_names = ()

class Print(Node):
    __slots__ = ('expr','coord')

    def __init__(self,expr=None,coord=None):
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None: nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    attr_names = ()

class Read(Node):
    __slots__ = ('expr','coord')

    def __init__(self,expr,coord=None):
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None: nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    attr_names = ()

class EmptyStatement(Node):
    __slots__ = ('coord')

    def __init__(self,coord=None):
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    attr_names = ()
