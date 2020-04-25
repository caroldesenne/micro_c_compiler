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
        for name in self.__slots__[:-1]:
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

class Coord(object):
    """ Coordinates of a syntactic element. Consists of:
            - Line number
            - (optional) column number, for the Lexer
    """
    __slots__ = ('line', 'column')

    def __init__(self, line, column=None):
        self.line = line
        self.column = column

    def __str__(self):
        if self.line:
            coord_str = "   @ %s:%s" % (self.line, self.column)
        else:
            coord_str = ""
        return coord_str

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
    __slots__ = ('decls','coord')

    def __init__(self, decl, coord=None):
        self.decls = decl
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
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

class FuncDecl(Node):
    __slots__ = ('args', 'type', 'coord')

    def __init__(self, args, type, coord=None):
        self.args = args
        self.type = type
        self.coord = coord

    def children(self):
        nodelist = []
        if self.args is not None: nodelist.append(("args", self.args))
        if self.type is not None: nodelist.append(("type", self.type))
        return tuple(nodelist)

    attr_names = ()

class FuncDef(Node):
    __slots__ = ('decl','type','decl_list','compound_statement','coord')

    def __init__(self, decl, type, dl, cs, coord=None):
        self.decl = decl
        self.type = type
        self.decl_list = dl
        self.compound_statement = cs
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        if self.decl is not None: nodelist.append(("decl", self.decl))
        if self.compound_statement is not None: nodelist.append(("compound_statement", self.compound_statement))
        for i, child in enumerate(self.decl_list or []):
            nodelist.append(("declaration[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()

class ParamList(Node):
    __slots__ = ('list', 'coord')

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
    __slots__ = ('type','size','coord')

    def __init__(self, type, s, coord=None):
        self.type = type
        self.size = s
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        if self.size is not None: nodelist.append(("size", self.size))
        return tuple(nodelist)

    attr_names = ()

class ArrayRef(Node):
    __slots__ = ('name','access_value','type','size','coord')

    def __init__(self, name, av, coord=None):
        self.name = name
        self.access_value = av
        self.type = None
        self.size = 0
        self.coord = coord

    def children(self):
        nodelist = []
        if self.name is not None: nodelist.append(("name", self.name))
        if self.access_value is not None: nodelist.append(("access_value", self.access_value))
        return tuple(nodelist)

    attr_names = ()

class FuncCall(Node):
    __slots__ = ('name','params','type','gen_location', 'coord')

    def __init__(self, name, params, coord=None):
        self.name = name
        self.params = params
        self.type = None
        self.gen_location = None
        self.coord = coord

    def children(self):
        nodelist = []
        if self.name is not None: nodelist.append(("name", self.name))
        if self.params is not None: nodelist.append(("params", self.params))
        return tuple(nodelist)

    attr_names = ()

class ID(Node):
    #__slots__ = ('name','type', 'gen_location', 'coord')
    __slots__ = ('name','type','coord')

    def __init__(self,name,coord=None):
        self.name = name
        self.type = None
        #self.gen_location = None
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    attr_names = ('name', )

class Assignment(Node):
    __slots__ = ('op','assignee','value','type','coord')

    def __init__(self,op,ass,v,coord=None):
        self.op = op
        self.assignee = ass
        self.value = v
        self.type = None
        self.coord = coord

    def children(self):
        nodelist = []
        if self.assignee is not None: nodelist.append(("assignee", self.assignee))
        if self.value is not None: nodelist.append(("value", self.value))
        return tuple(nodelist)

    attr_names = ('op', )

class PtrDecl(Node):
    __slots__ = ('stars', 'type', 'coord')

    def __init__(self, stars, coord=None):
        self.stars = stars
        self.type = None
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        return tuple(nodelist)

    attr_names = ()

class BinaryOp(Node):
    __slots__ = ('op', 'left', 'right', 'type', 'gen_location', 'coord')

    def __init__(self, op, left, right, coord=None):
        self.op = op
        self.left = left
        self.right = right
        self.type = type
        self.gen_location = None
        self.coord = coord

    def children(self):
        nodelist = []
        if self.left is not None: nodelist.append(("left", self.left))
        if self.right is not None: nodelist.append(("right", self.right))
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
    __slots__ = ('op','expression','type','coord')

    def __init__(self, op, exp, coord=None):
        self.op = op
        self.expression = exp
        self.type = None
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expression is not None: nodelist.append(("expression", self.expression))
        return tuple(nodelist)

    attr_names = ('op', )

class Constant(Node):
    __slots__ = ('type', 'value', 'size', 'gen_location', 'coord')

    def __init__(self, type, value, coord=None):
        self.type = type
        self.value = value
        self.coord = coord
        self.gen_location = None
        if(self.type=='string'):
            self.size = len(value)-2
        else:
            self.size = 1

    def children(self):
        nodelist = []
        return tuple(nodelist)

    attr_names = ('type', 'value', )

class Type(Node):
    __slots__ = ('names','arrayLevel','coord')

    def __init__(self, names, coord=None):
        self.names = names
        self.arrayLevel = 0 # int a;    : arrayLevel = 0
                            # int b[]   : arrayLevel = 1
                            # int c[][] : arrayLevel = 2
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    attr_names = ('names', )

class VarDecl(Node):
    __slots__ = ('name','type','coord')

    def __init__(self, name, coord=None):
        self.name = name
        self.type = None
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        return tuple(nodelist)

    attr_names = ()

class Decl(Node):
    __slots__ = ('name', 'type', 'init', 'isFunction','coord')

    def __init__(self, name, type, init, coord=None):
        self.name = name
        self.type = type
        self.init = init
        self.isFunction = False
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        if self.init is not None: nodelist.append(("init", self.init))
        return tuple(nodelist)

    attr_names = ('name', )

class InitList(Node):
    __slots__ = ('inits','type','size','coord')

    def __init__(self, inits, coord=None):
        self.inits = inits
        self.type = None
        self.size = len(inits)
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.inits or []):
            nodelist.append(("initializer[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()

class Compound(Node):
    __slots__ = ('block_items','coord')

    def __init__(self, bi, coord=None):
        self.block_items = bi
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.block_items or []):
            nodelist.append(("block_items[%d]" % i, child))
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
    __slots__ = ('list','type','coord')

    def __init__(self,list,coord=None):
        self.list = list
        self.type = None
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
    __slots__ = ('expr','type','coord')

    def __init__(self,expr=None,coord=None):
        self.expr = expr
        self.type = None
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
