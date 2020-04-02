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

class NodeVisitor(object):
	""" A base NodeVisitor class for visiting uc_ast nodes.
		Subclass it and define your own visit_XXX methods, where
		XXX is the class name you want to visit with these
		methods.

		For example:

		class ConstantVisitor(NodeVisitor):
			def __init__(self):
				self.values = []

			def visit_Constant(self, node):
				self.values.append(node.value)

		Creates a list of values of all the constant nodes
		encountered below the given node. To use it:

		cv = ConstantVisitor()
		cv.visit(node)

		Notes:

		*   generic_visit() will be called for AST nodes for which
			no visit_XXX method was defined.
		*   The children of nodes for which a visit_XXX was
			defined will not be visited - if you need this, call
			generic_visit() on the node.
			You can use:
				NodeVisitor.generic_visit(self, node)
		*   Modeled after Python's own AST visiting facilities
			(the ast module of Python 3.0)
	"""

	_method_cache = None

	def visit(self, node):
		""" Visit a node.
		"""

		if self._method_cache is None:
			self._method_cache = {}

		visitor = self._method_cache.get(node.__class__.__name__, None)
		if visitor is None:
			method = 'visit_' + node.__class__.__name__
			visitor = getattr(self, method, self.generic_visit)
			self._method_cache[node.__class__.__name__] = visitor

		return visitor(node)

	def generic_visit(self, node):
		""" Called if no explicit visitor function exists for a
			node. Implements preorder visiting of the node.
		"""
		for c in node:
			self.visit(c)

'''
TODO:
ArrayDecl ( ), ArrayRef ( ), Assert ( ), Assignment (op), Break ( ), 
Cast ( ), Compound ( ), Decl (name), DeclList ( ), EmptyStatement ( ), 
ExprList ( ), For ( ), FuncCall ( ), FuncDecl ( ), 
If ( ), InitList ( ), ParamList ( ), Print ( ), PtrDecl ( ), 
Return ( ), Type (names), VarDecl (), UnaryOp (op), While ( )
'''

'''
DONE:
Program ( )
GlobalDecl ( )
FuncDef ( )
ID (name)

BinaryOp (op)

Constant (type, value)
Read ( )
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

class ID(Node):
	__slots__ = ('name','coord')

	def __init__(self,name,coord=None):
		self.name = name

	def children(self):
		nodelist = []
		return tuple(nodelist)

	attr_names = ('name', )

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

