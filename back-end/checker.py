import ply.yacc as yacc
#import uctype
from parser import Parser
from ast import *

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
                    as_ops   = {'=','+=','-=','*=','/='}
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
                   bool_ops = {'==','!='},
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
pointer_type = uCType(name = "pointer",
                      bin_ops  = {},
                      un_ops   = {},
                      bool_ops = {},
                      as_ops   = {}
                     )

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
        self.add("int",int_type)
        self.add("float",float_type)
        self.add("char",char_type)
        self.add("string",string_type)
        self.add("bool",boolean_type)
        self.add("array",array_type)
        self.add("pointer",pointer_type)

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

def getArrayName(node):
    inner = node
    while not isinstance(inner,VarDecl):
        inner = inner.type
    return inner.name.name

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
        self.declarations = []

    def checkArgumentsMatch(self,args1,args2,coord):
        err = f"{coord.line}:{coord.column} - argument list sizes must be the same."
        assert (args1==None and args2==None) or (len(args1)==len(args2)), err
        for i, arg in enumerate(args1 or []):
            pbt = getBasicType(arg)
            abt = getBasicType(args2[i])
            err = f"{arg.coord.line}:{arg.coord.column} - arguments types must match: expected {pbt}, got {abt}."
            pt = getInnerMostType(arg)
            at = getInnerMostType(args2[i])
            assert typesEqual(pt,at), err

    def visit_Program(self,node):
        self.scopes.pushLevel()
        for i, child in enumerate(node.gdecls or []):
            self.visit(child)
        self.scopes.popLevel()

    def visit_GlobalDecl(self,node):
        for i, child in enumerate(node.decls or []):
            self.visit(child)

    def copyDeclList(self,node):
        node.decl_list = []
        for d in self.declarations:
            node.decl_list.append(d)
        self.declarations = []

    def visit_FuncDef(self,node):
        self.visit(node.type)
        decl = node.decl
        while not isinstance(decl,Decl):
            decl = decl.type
        decl.isPrototype = False # new scope is pushed in visit_Decl()
        self.visit(node.decl)
        # get parameter list from node.decl
        node.param_list = node.decl.type.args
        # insert in scope expected return type
        self.scopes.insert("return",node.type)
        #for i, child in enumerate(node.decl_list or []):
        #    self.visit(child)
        self.declarations = []
        self.visit(node.compound_statement)
        self.scopes.popLevel()
        self.copyDeclList(node)

    #######################################################################
    # From here till Decl, all these functions are used for Decl purposes #
    #######################################################################
    def visit_Prototype(self,node): # this is the prototype case (FuncDecl outside a FuncDef)
        sym = node.name.name
        alreadyDefined = f"{node.coord.line}:{node.coord.column} - symbol {sym} already defined in current scope."
        assert self.scopes.insert(sym,node.type), alreadyDefined
        self.scopes.pushLevel()
        self.visit(node.type)
        self.scopes.popLevel()

    def visit_FuncDefinition(self,node): # this is not a prototype
        sym = node.name.name
        alreadyDefined = f"{node.coord.line}:{node.coord.column} - symbol {sym} already defined in current scope."
        proto = self.scopes.find(sym)
        if proto: # check if there is some prototype already defined
            assert proto.isPrototype, alreadyDefined # this proto is actually another function, not a prototype
            node.type.proto = proto # keep this value here for parameters and type checks later
        self.scopes.insert(sym,node.type)
        self.scopes.pushLevel()
        self.visit(node.type)

    def visit_DeclFuncDecl(self,node):
        node.type.isPrototype = node.isPrototype
        if node.isPrototype:
            self.visit_Prototype(node)
        else:
            self.visit_FuncDefinition(node)

    def visit_DeclVarOrArray(self,node):
        sym = node.name.name # get ID from Decl (which is called name), then its name
        alreadyDefined = f"{node.coord.line}:{node.coord.column} - symbol {sym} already defined in current scope."
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
                tdb = getBasicType(node)
                tib = getBasicType(node.init)
                err = f"{node.coord.line}:{node.coord.column} - declaration and initializer types must match: expected {tdb}, got {tib}."
                assert typesEqual(td,ti), err

    def visit_Decl(self,node):
        self.declarations.append(node)
        if isinstance(node.type,FuncDecl):
            self.visit_DeclFuncDecl(node)
        elif isinstance(node.type,PtrDecl):
            assert False, "PtrDecl not implemented yet."
        else:
            self.visit_DeclVarOrArray(node)

    def check_Decl_ArrayDecl(self,node):
        isString = False
        if getBasicType(node)=='char' and getBasicType(node.init)=='string':
            isString = True
            c = Constant(type='char', value=node.init.size)
            self.visit(c)
            node.type.size = [c]
        # check init type (must be InitList)
        err = f"{node.coord.line}:{node.coord.column} - array initializer must be of array type."
        assert isString or isinstance(node.init,InitList) or isinstance(node.init,ArrayRef), err
        # check if their sizes match
        ad = node.type
        # test if initialization matches array sizes
        if ad.size:
            err = f"{node.coord.line}:{node.coord.column} - array initializer size must match declaration."
            init = node.init
            i = 0
            while isinstance(init, InitList):
                assert int(ad.size[i].value)==int(init.size), err
                i += 1
                init = init.inits
        # if size wasnt specified, set it
        if ad.size==[]:
            init = node.init
            while isinstance(init, InitList):
                c = Constant(type='int', value=init.size)
                self.visit(c)
                ad.size.append(c)
                init = init.inits[0]
        if isinstance(node.init,InitList):
            node.init.sizes = ad.size
    #######################################################################
    #             Here ends the functions used for Decl purposes          #
    #######################################################################

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
        rbt = getBasicType(t)
        nt = getInnerMostType(node)
        nbt = getBasicType(node)
        err = f"{node.coord.line}:{node.coord.column} - wrong return type in function: expected {rbt}, got {nbt}."
        assert typesEqual(t,nt), err

    def visit_ArrayDecl(self,node):
        if self.scopes.depth==1:
            node.isGlobal = True
        self.visit(node.type)
        t = getInnerMostType(node)
        t.arrayLevel += 1        
        # if there is a size, put it on a list and concatenate it with node.type.size (inner size)
        if node.size:
            self.visit(node.size)
            node.size = [node.size]+node.type.size
        else:
            node.size = []

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
        bnt = getBasicType(node)
        for i, child in enumerate(node.inits or []):
            self.visit(child)
            ct = getInnerMostType(child)
            bct = getBasicType(child)
            assert typesEqual(ct,nt), f"{child.coord.line}:{child.coord.column} - types in initializer list must all match: expected {bnt}, got {bct}."

    def visit_FuncDecl(self,node):
        if node.args:
            self.visit(node.args)
        self.visit(node.type)
        if node.proto: # check if everything matches with prototype
            # check matching types
            bt = getBasicType(node)
            pbt = getBasicType(node.proto)
            tmatch = f"{node.coord.line}:{node.coord.column} - prototype and function definition types must match: expected {pbt}, got {bt}."
            assert bt==pbt, tmatch
            # check parameter types and size
            self.checkArgumentsMatch(node.proto.args.list,node.args.list,node.coord)

    def visit_ParamList(self,node):
        for i, child in enumerate(node.list or []):
            self.visit(child)

    def visit_Assignment(self,node):
        self.visit(node.assignee)
        # Check that the types match
        self.visit(node.value)
        node.type = node.assignee.type
        at = getInnerMostType(node.assignee)
        bat = getBasicType(node.assignee)
        vt = getInnerMostType(node.value)
        bvt = getBasicType(node.value)
        err = f"{node.coord.line}:{node.coord.column} - type mismatch in assignment: expected {bat}, got {bvt}."
        if bat==bvt and isinstance(node.assignee, ArrayRef):
            err = f"{node.coord.line}:{node.coord.column} - bad type for array reference."
        assert typesEqual(at,vt), err

    def visit_Assert(self,node):
        self.visit(node.expr)
        bt = getBasicType(node.expr)
        err = f"{node.coord.line}:{node.coord.column} - assert expression mismacth: expected bool, got {bt}."
        assert bt=='bool', err

    def visit_FuncCall(self,node):
        self.visit(node.params)
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
        # check arguments types match parameters type and list size matches declaration
        self.checkArgumentsMatch(arg_list,par_list,node.coord)

    def visit_Constant(self,node):
        if not isinstance(node.type,Type):
            bt = node.type
            node.type = Type(names=[bt])

    def visit_BinaryOp(self,node):
        self.visit(node.left)
        self.visit(node.right)
        # Make sure left and right operands have the same type
        tl = getInnerMostType(node.left)
        tr = getInnerMostType(node.right)
        btl = getBasicType(node.left)
        btr = getBasicType(node.right)
        binop = f"{node.coord.line}:{node.coord.column} - left and right sides of binary operation types mismatch: {btl} (left) and {btr} (right)."
        assert typesEqual(tl,tr), binop
        # assign left type to node's type
        node.type = node.left.type
        gt = getInnerMostType(node)
        bt = getBasicType(node)
        if gt.arrayLevel > 0:
            bt = "array"
        t = self.scopes.find(bt)
        # check if type exists
        assert t != None, f"{node.coord.line}:{node.coord.column} - non supported type {bt}."
        # make sure the operation is supported for this type
        assert node.op in t.bin_ops, f"{node.coord.line}:{node.coord.column} - {node.op} binary operation not supported for type {bt}."
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
        assert t != None, f"{node.coord.line}:{node.coord.column} - non supported type {bt}."
        # make sure the operation is supported for this type
        assert node.op in t.un_ops, f"{node.coord.line}:{node.coord.column} - {node.op} unary operation not supported for type {bt}."

    def visit_Print(self,node):
        if node.expr:
            self.visit(node.expr)

    def visit_ArrayRef(self,node):
        self.visit(node.name)
        self.visit(node.access_value)
        if node.size:
            self.visit(node.size)
        # check if this is an array
        at = getInnerMostType(node.name)
        err = f"{node.coord.line}:{node.coord.column} - array reference to non array variable."
        assert at.arrayLevel > 0, err
        # check if there is an access value
        accNone = f"{node.coord.line}:{node.coord.column} - array access value must be specified."
        assert node.access_value, accNone
        # check if the access value is an integer
        bt = getBasicType(node.access_value)
        accInt = f"{node.coord.line}:{node.coord.column} - array access value type mismatch: expected int, got {bt}."
        assert bt=="int", accInt
        # take same type as this array with an array level lower
        node.type = Type(at.names)
        node.type.arrayLevel = at.arrayLevel-1
        # get element size
        if node.type.arrayLevel > 0:
            innarray = node.name.type
            node.size = innarray.size[1:]
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
            # TODO copy
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
        assert t, notDefined
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
