import ply.yacc as yacc
import uctype
from parser import Parser
from collections import defaultdict
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

def getInnerMostType(node):
    td = node
    while td and (not isinstance(td,Type)):
        td = td.type
    return td

def getBasicType(node):
    t = getInnerMostType(node)
    return t.names[0]

class GenerateCode(NodeVisitor):
    '''
    Node visitor class that creates 3-address encoded instruction sequences.
    '''
    def __init__(self):
        super(GenerateCode, self).__init__()

        # version dictionary for temporaries
        self.versions = defaultdict(int)

        # The generated code (list of tuples)
        self.code = []

        self.temp_var_dict = {}

    def new_temp(self, typeobj):
        '''
        Create a new temporary variable of a given type.
        '''
        name = "t_%d" % (self.versions[typeobj.names[0]])
        self.versions[typeobj.names[0]] += 1
        return name

    # You must implement visit_Nodename methods for all of the other
    # AST nodes.  In your code, you will need to make instructions
    # and append them to the self.code list.
    #
    # A few sample methods follow.  You may have to adjust depending
    # on the names of the AST nodes you've defined.

    def visit_Program(self, node):
        for i, child in enumerate(node.gdecls or []):
            self.visit(child)

    def visit_FuncDef(self, node):
        self.code.append(('define', '_{}'.format(node.decl.name.name)))
        # for i, child in enumerate(node.decl_list or []):
        #     self.visit(child)

        self.visit(node.compound_statement)
        self.code.append(('end', ''))

    # def visit_FuncDecl(self, node):



    def visit_Compound(self, node):
        for i, child in enumerate(node.block_items or []):
            self.visit(child)

    def visit_Decl(self, node):
        target = self.new_temp(node.type.type)
        self.temp_var_dict[node.name.name] = target

        # Make the SSA opcode and append to list of generated instructions
        inst = ('alloc_' + getBasicType(node.type), target)
        self.code.append(inst)

        if node.init:
            self.visit(node.init)

            # target_store = self.new_temp(node.type.type)
            self.code.append(('store_' + getBasicType(node), node.init.gen_location, target))

    def visit_Return(self, node):
        # Create a new temporary variable name
        target = self.new_temp(node.expr.type)

        self.visit(node.expr)

        self.code.append(('store_' + getBasicType(node.expr), node.expr.gen_location, target))

        # Make the SSA opcode and append to list of generated instructions
        inst = ('return_' + getBasicType(node), target)
        self.code.append(inst)

    def visit_Constant(self, node):
        # Create a new temporary variable name
        target = self.new_temp(node.type)

        # Make the SSA opcode and append to list of generated instructions
        inst = ('literal_'+node.type.names[0], node.value, target)
        self.code.append(inst)

        # Save the name of the temporary variable where the value was placed
        node.gen_location = target

    def visit_ID(self, node):
        node.gen_location = self.temp_var_dict[node.name]
        return self.new_temp(node.type.type)

    def visit_BinaryOp(self, node):
        # Visit the left and right expressions
        self.visit(node.left)
        self.visit(node.right)

        # Make a new temporary for storing the result
        target = self.new_temp(getInnerMostType(node.type))

        # Create the opcode and append to list
        # opcode = binary_ops[node.op] + "_"+node.left.type.name #TODO create op conv table * --> mul
        opcode = node.op + "_"+getBasicType(node.left)
        inst = (opcode, node.left.gen_location, node.right.gen_location, target)
        self.code.append(inst)

        # Store location of the result on the node
        node.gen_location = target

    def visit_PrintStatement(self, node):
        # Visit the expression
        self.visit(node.expr)

        # Create the opcode and append to list
        inst = ('print_'+node.expr.type.name, node.expr.gen_location)
        self.code.append(inst)

    def visit_VarDeclaration(self, node):
        # allocate on stack memory
        inst = ('alloc_'+node.type.name,
                    node.id)
        self.code.append(inst)
        # store optional init val
        if node.value:
            self.visit(node.value)
            inst = ('store_'+node.type.name,
                    node.value.gen_location,
                    node.id)
            self.code.append(inst)

    def visit_LoadLocation(self, node):
        target = self.new_temp(node.type)
        inst = ('load_'+node.type.name,
                node.name,
                target)
        self.code.append(inst)
        node.gen_location = target

    def visit_AssignmentStatement(self, node):
        self.visit(node.value)
        inst = ('store_'+node.value.type.name,
                node.value.gen_location,
                node.location)
        self.code.append(inst)

    def visit_UnaryOp(self, node):
        self.visit(node.left)
        target = self.new_temp(node.type)
        opcode = unary_ops[node.op] + "_" + node.left.type.name
        inst = (opcode, node.left.gen_location)
        self.code.append(inst)
        node.gen_location = target
