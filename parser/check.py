import ply.yacc as yacc
import uctype
from parser import Parser
from uc_ast import *

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

class SymbolTable(object):
    '''
    Class representing a symbol table.  It should provide functionality
    for adding and looking up nodes associated with identifiers.
    '''
    def __init__(self):
        self.symtab = {}
        # Add built-in type names (int, float, char) to every symbol table
        self.add("int",uctype.int_type)
        self.add("float",uctype.float_type)
        self.add("char",uctype.char_type)
        # self.symtab.add("bool",uctype.boolean_type) TODO: there is no bool

    def lookup(self, a):
        return self.symtab.get(a)

    def add(self, a, v):
        self.symtab[a] = v

class Scopes(object):
    '''
    Class representing all the scopes in a program. Each scope level is
    represented by a symbol table, and they are assembled in a list. The first
    element of the list is the root scope and we go into deeper scopes as we
    go through the array. Depth represents the maximal scope depth we are in
    at the moment (and it corresponds to the scopes array size).
    '''
    def __init__(self):
        root = SymbolTable()
        self.scope = [root]
        self.depth = 1

    def pushLevel(self):
        s = SymbolTable()
        self.scope.append(s)
        self.depth += 1

    def popLevel(self):
        self.scope.pop()
        self.depth -= 1

    def insert(self,sym,t):
        '''
        insert inserts a new symbol in the symbol table of the current scope (which is
        the one represented by depth, or self.scope[self.depth-1]). If the symbol already exists
        in the current scope, return false. Otherwise, insert it and return true.
        '''
        s = self.scope[self.depth-1]
        if s.lookup(sym) == None:
            s.add(sym,t)
            return True
        return False

    def find(self,sym):
        '''
        find tries to find a symbol in the current scope or in previous scopes. If it finds it,
        return the corresponding found, otherwise returns None.
        '''
        currentDepth = self.depth
        while(currentDepth > 0):
            s = self.scope[currentDepth-1]
            if s.lookup(sym) != None:
                return s.lookup(sym)
            currentDepth -= 1
        return None

'''
a * means we still have TODO in visit for this class

FuncDecl ( ) *
Program ( ) *
GlobalDecl ( ) *
DeclList ( ) *
ArrayDecl ( ) *
ArrayRef ( ) *
VarDecl () *
ID (name) *
Assignment (op) *
PtrDecl ( ) *
FuncCall ( ) *
FuncDef ( ) *
BinaryOp (op) *
UnaryOp (op) *
Constant (type, value) *
Type (names) *
Decl (name) *
InitList ( ) *
ExprList ( ) *
Compound ( ) *
If ( ) *
While ( ) *
For ( ) *
Break ( ) *
Return ( ) *
Assert ( ) *
Print ( ) *
Read ( ) *
EmptyStatement ( ) *
'''

class CheckProgramVisitor(NodeVisitor):
    '''
    Program checking class. This class uses the visitor pattern. You need to define methods
    of the form visit_NodeName() for each kind of AST node that you want to process.
    Note: You will need to adjust the names of the AST nodes if you picked different names.
    '''
    def __init__(self):
        # Initialize the symbol table
        self.symtab = SymbolTable()

    def visit_Program(self,node):
        # 1. Visit all of the statements
        # 2. Record the associated symbol table
        print("----------hey I got to visit_Program!!!-----------")
        # TODO
        for i, child in enumerate(node.gdecls or []):
            self.visit(child)

    def visit_GlobalDecl(self,node):
        # TODO
        print("----------hey I got to visit_GlobalDecl!!!-----------")

    def visit_FuncDef(self,node):
        # TODO
        print("----------hey I got to visit_FuncDef!!!-----------")

    def visit_BinaryOp(self, node):
        # 1. Make sure left and right operands have the same type
        # 2. Make sure the operation is supported
        # 3. Assign the result type
        self.visit(node.left)
        self.visit(node.right)
        node.type = node.left.type

    def visit_Assignment(self,node):
        ## 1. Make sure the location of the assignment is defined
        # sym = self.symtab.lookup(node.location) TODO FIX THIS
        assert sym, "Assigning to unknown sym"
        ## 2. Check that the types match
        self.visit(node.value)
        assert sym.type == node.value.type, "Type mismatch in assignment"


def scopesTest():
    sc = Scopes()
    print(sc.insert("hello",2)) # true
    sc.pushLevel()
    print(sc.insert("cat",3)) # true
    print(sc.insert("hello",1)) # true
    print(sc.find("cat")) # 3
    print(sc.find("hello")) # 1
    sc.popLevel()
    print(sc.insert("hello",1)) # false
    print(sc.find("hello")) # 2

if __name__ == '__main__':

    import sys

    p = Parser()
    code = open(sys.argv[1]).read()
    ast = p.parse(code)
    #scopesTest()

    #ast.show()
    #check = CheckProgramVisitor()
    #check.visit_Program(ast)

