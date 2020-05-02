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
        self.add("string",uctype.string_type)
        self.add("bool",uctype.boolean_type)
        self.add("array",uctype.array_type)
        self.add("pointer",uctype.pointer_type)

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
        self.scope = []
        self.loopStack = []
        self.depth = 0

    def pushLoop(self,loop):
        self.loopStack.append(loop)

    def popLoop(self):
        if not self.inLoop():
            return None
        return self.loopStack.pop()

    def inLoop(self):
        if len(self.loopStack) == 0:
            return False
        return True

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
            if s.lookup(sym):
                return s.lookup(sym)
            currentDepth -= 1
        return None

def getInnerMostType(node):
    td = node
    while td and (not isinstance(td,Type)):
        td = td.type
    return td

def getBasicType(node):
    t = getInnerMostType(node)
    return t.names[0]

def typesEqual(t1,t2):
    return t1.names==t2.names and t1.arrayLevel==t2.arrayLevel

class CheckProgramVisitor(NodeVisitor):
    '''
    Program checking class. This class uses the visitor pattern. You need to define methods
    of the form visit_NodeName() for each kind of AST node that you want to process.
    Note: You will need to adjust the names of the AST nodes if you picked different names.
    '''
    def __init__(self):
        self.scopes = Scopes()

    def visit_Program(self,node):
        self.scopes.pushLevel()
        for i, child in enumerate(node.gdecls or []):
            self.visit(child)
        self.scopes.popLevel()

    def visit_GlobalDecl(self,node):
        for i, child in enumerate(node.decls or []):
            self.visit(child)

    def visit_FuncDef(self,node):
        self.visit(node.type)

        decl = node.decl
        while not isinstance(decl,Decl):
            decl = decl.type
        decl.isPrototype = False # new scope is pushed in visit_Decl()
        self.visit(node.decl)
        # insert in scope expected return type
        self.scopes.insert("return",node.type)
        for i, child in enumerate(node.decl_list or []):
            self.visit(child)
        self.visit(node.compound_statement)
        self.scopes.popLevel()

    def visit_Decl(self,node):
        sym = node.name.name # get ID from Decl (which is called name), then its name
        # check if symbol exists already, otherwise insert it in scope
        alreadyDefined = f"{node.coord.line}:{node.coord.column} - symbol {sym} already defined in current scope."

        if isinstance(node.type,FuncDecl):
            # check names and parameter types
            node.type.isPrototype = node.isPrototype
            if node.isPrototype: # this is the prototype case (FuncDecl outside a FuncDef)
                assert self.scopes.insert(sym,node.type), alreadyDefined
            else:
                proto = self.scopes.find(sym)
                if proto: # check if there is some prototype already defined
                    assert proto.isPrototype, alreadyDefined # this proto is actually another function, not prototype
                    # TODO test if prototype and function params match
                    
                else:
                    self.scopes.insert(sym,node.type)
            # visit and deal with levels
            self.scopes.pushLevel()
            self.visit(node.type)
            if node.isPrototype: # this is the prototype case (FuncDecl outside a FuncDef)
                self.scopes.popLevel()
            return

        if isinstance(node.type,PtrDecl):
            assert False, "PtrDecl not implemented yet."
        
        assert self.scopes.insert(sym,node.type), alreadyDefined
        self.visit(node.type)

        if node.init:
            self.visit(node.init)
            if isinstance(node.type,ArrayDecl):
                self.check_Decl_ArrayDecl(node)
            else:
                # check if declaration type matches initializer type
                td = getInnerMostType(node)
                ti = getInnerMostType(node.init)
                assert typesEqual(td,ti), f"{node.coord.line}:{node.coord.column} - declaration and initializer types must match."

    def check_Decl_ArrayDecl(self,node):
        isString = False
        if getBasicType(node)=='char' and getBasicType(node.init)=='string':
            isString = True
        # check init type (must be InitList)
        err = f"{node.coord.line}:{node.coord.column} - array initializer must be of array type."
        assert isString or isinstance(node.init,InitList) or isinstance(node.init,ArrayRef), err
        # check if their sizes match
        ad = node.type
        if ad.size:
            err = f"{node.coord.line}:{node.coord.column} - array initializer size must match declaration."
            assert int(ad.size.value)==int(node.init.size), err
        # if size wasnt specified, set it
        if ad.size==None:
            ad.size = Constant(type='int', value=node.init.size)

    def visit_Compound(self,node):
        for i, child in enumerate(node.block_items or []):
            self.visit(child)

    def visit_For(self,node):
        self.scopes.pushLevel()
        self.scopes.pushLoop('for')
        if node.init:
            self.visit(node.init)
        if node.stop_cond:
            self.visit(node.stop_cond)
            mustBool = f"{node.coord.line}:{node.coord.column} - stop condition must be of type bool."
            bt = getBasicType(node.stop_cond)
            assert bt=="bool", mustBool
        if node.increment:
            self.visit(node.increment)
        if node.statement:
            self.visit(node.statement)
        self.scopes.popLoop()
        self.scopes.popLevel()

    def visit_While(self,node):
        self.scopes.pushLoop('while')
        self.visit(node.cond)
        bt = getBasicType(node.cond)
        assert bt=='bool', f"{node.coord.line}:{node.coord.column} - while condition must be of type bool."
        if node.statement:
            self.visit(node.statement)
        self.scopes.popLoop()

    def visit_Break(self,node):
        err = f"{node.coord.line}:{node.coord.column} - break statement should be inside a loop."
        assert self.scopes.inLoop(), err

    def visit_DeclList(self,node):
        for i, child in enumerate(node.decls or []):
            self.visit(child)

    def visit_Return(self,node):
        if node.expr:
            self.visit(node.expr)
            node.type = node.expr.type
        else:
            node.type = Type(names=['void'])
        # check return type matches function definition
        t = self.scopes.find("return")
        bt = getInnerMostType(node)
        err = f"{node.coord.line}:{node.coord.column} - wrong return type in function: expected {t}, got {bt}."
        assert typesEqual(t,bt), err

    def visit_ArrayDecl(self,node):
        if self.scopes.depth==1:
            node.isGlobal = True
        self.visit(node.type)
        t = getInnerMostType(node)
        t.arrayLevel += 1

    def visit_VarDecl(self,node):
        if self.scopes.depth==1:
            node.isGlobal = True
        self.visit(node.type)

    def visit_Type(self,node):
        pass

    def visit_InitList(self,node):
        if node.inits and len(node.inits)>0:
            self.visit(node.inits[0])
            node.type = node.inits[0].type
        nt = getInnerMostType(node)
        for i, child in enumerate(node.inits or []):
            self.visit(child)
            ct = getInnerMostType(child)
            assert typesEqual(ct,nt), f"{child.coord.line}:{child.coord.column} - types in initializer list must all match."

    def visit_FuncDecl(self,node):
        if node.args:
            self.visit(node.args)
        self.visit(node.type)

    def visit_ParamList(self,node):
        for i, child in enumerate(node.list or []):
            self.visit(child)

    def visit_Assignment(self,node):
        self.visit(node.assignee)
        # Check that the types match
        self.visit(node.value)
        node.type = node.assignee.type
        err = f"{node.coord.line}:{node.coord.column} - type mismatch in assignment: ({node.assignee.type},{node.value.type})."
        at = getInnerMostType(node.assignee)
        vt = getInnerMostType(node.value)
        assert typesEqual(at,vt), err

    def visit_Assert(self,node):
        self.visit(node.expr)
        err = f"{node.coord.line}:{node.coord.column} - assert expression must be of type bool, got {node.expr.type}."
        bt = getBasicType(node.expr)
        assert bt=='bool', err

    def visit_FuncCall(self,node):
        self.visit(node.name)
        node.type = node.name.type
        func = self.scopes.find(node.name.name)
        # check function declaration
        assert func, f"{node.coord.line}:{node.coord.column} - undeclared function."
        assert isinstance(func,FuncDecl), f"{node.coord.line}:{node.coord.column} - cannot call non-function symbol {node.name.name}."
        # check size of parameters list
        arg_list = func.args.list
        if not isinstance(node.params,ExprList):
            par_list = [node.params]
        else:
            par_list = node.params.list
        err = f"{node.coord.line}:{node.coord.column} - number of arguments must match with function declaration."
        assert len(arg_list)==len(par_list), err
        # check arguments types match parameters type
        for i, par in enumerate(par_list or []):
            self.visit(par)
            err = f"{par.coord.line}:{par.coord.column} - argument type doesn't match parameter in function declaration."
            pt = getInnerMostType(par)
            at = getInnerMostType(arg_list[i])
            assert typesEqual(pt,at), err

    def visit_Constant(self,node):
        if not isinstance(node.type,Type):
            bt = node.type
            node.type = Type(names=[bt])

    def visit_BinaryOp(self,node):
        self.visit(node.left)
        self.visit(node.right)
        # Make sure left and right operands have the same type
        binop = f"{node.coord.line}:{node.coord.column} - left ({node.left.type}) and right ({node.right.type}) sides of binary operation must have the same types."
        btr = getInnerMostType(node.left)
        btl = getInnerMostType(node.right)
        assert typesEqual(btr,btl), binop
        # assign left type to node's type
        node.type = node.left.type
        gt = getInnerMostType(node)
        bt = getBasicType(node)
        if gt.arrayLevel > 0:
            bt = "array"
        t = self.scopes.find(bt)
        # check if type exists
        assert t != None, f"{node.coord.line}:{node.coord.column} - inexistent type {node.type}."
        # make sure the operation is supported for this type
        assert node.op in t.bin_ops, f"{node.coord.line}:{node.coord.column} - {node.op} binary operation not supported for type {node.type}."
        # verify if we are making a relational operation and assign bool type
        if node.op in t.bool_ops:
            node.type = Type(names=['bool'])

    def visit_UnaryOp(self,node):
        self.visit(node.expression)
        # assing the result type (check which operation it is)
        node.type = node.expression.type
        gt = getInnerMostType(node)
        bt = getBasicType(node)
        if gt.arrayLevel > 0:
            bt = "array"
        t = self.scopes.find(bt)
        # check if type exists
        assert t != None, f"{node.coord.line}:{node.coord.column} - inexistent type {node.type}."
        # make sure the operation is supported for this type
        assert node.op in t.un_ops, f"{node.coord.line}:{node.coord.column} - {node.op} unary operation not supported for type {node.type}."

    def visit_Print(self,node):
        if node.expr:
            self.visit(node.expr)

    def visit_ArrayRef(self,node):
        self.visit(node.name)
        self.visit(node.access_value)
        # check if this is an array
        at = getInnerMostType(node.name)
        f"{node.coord.line}:{node.coord.column} - array reference to non array variable."
        assert at.arrayLevel > 0, err
        # check if there is an access value
        accNone = f"{node.coord.line}:{node.coord.column} - array access value must be specified."
        assert node.access_value, accNone
        # check if the access value is an integer
        accInt = f"{node.coord.line}:{node.coord.column} - array access value must be of type int."
        bt = getBasicType(node.access_value)
        assert bt=="int", accInt
        # take same type as this array with an array level lower
        node.type = Type(at.names)
        node.type.arrayLevel = at.arrayLevel-1
        # get element size        
        if node.type.arrayLevel > 0:
            innarray = node.name.type.type
            size = innarray.size.value
            node.size = size
        else:
            node.size = 1

    def visit_If(self,node):
        self.visit(node.cond)
        bt = getBasicType(node.cond)
        assert bt=='bool', f"{node.coord.line}:{node.coord.column} - if condition must be of type bool."
        if node.statement:
            self.visit(node.statement)
        # else new scope
        if node.else_st:
            self.visit(node.else_st)

    def visit_ExprList(self,node):
        for i, child in enumerate(node.list or []):
            self.visit(child)
        if node.list:
            node.type = node.list[0].type

    def visit_Read(self,node):
        if node.expr:
            self.visit(node.expr)

    def visit_Cast(self,node):
        if node.expression:
            self.visit(node.expression)
        t = getBasicType(node.type)
        assert t != None, f"{node.coord.line}:{node.coord.column} - must specify cast type."

    def visit_PtrDecl(self,node):
        node.type = Type(names=['pointer'])

    def visit_ID(self,node):
        sym = node.name
        t = self.scopes.find(sym)
        # check if symbol exists
        notDefined = f"{node.coord.line}:{node.coord.column} - symbol {sym} not defined."
        assert sym, notDefined
        node.type = t

    def visit_EmptyStatement(self,node):
        pass

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

    code = open(sys.argv[1]).read()
    # parse code and generate AST
    p = Parser()
    ast = p.parse(code)
    # perform semantic checks
    check = CheckProgramVisitor()
    check.visit_Program(ast)
