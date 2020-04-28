import sys
import ply.yacc as yacc
import uctype
from pprint import pprint
from parser import Parser
from collections import defaultdict
from uc_ast import *
from check import *

binary_ops = {
    '+': 'add',
    '*': 'mul',
    '-': 'sub',
    '/': 'div',
    '%': 'mod',
}

unary_ops = {
    '+': 'uadd',
    '-': 'uneg',
}

'''
TODO:

ArrayDecl
ArrayRef
Assert
Break
DeclList
EmptyStatement
ExprList
For
If
InitList
Print
PtrDecl
Read
Type
While

'''

class GenerateCode(NodeVisitor):
    '''
    Node visitor class that creates 3-address encoded instruction sequences.
    '''
    def __init__(self):
        super(GenerateCode, self).__init__()

        # version dictionary for temporaries
        self.fname = 'main'  # We use the function name as a key
        self.versions = {self.fname:0}

        # The generated code (list of tuples)
        self.code = []

        self.temp_var_dict = {}

    def new_temp(self):
        '''
        Create a new temporary variable of a given scope (function name).
        '''
        if self.fname not in self.versions:
            self.versions[self.fname] = 0
        name = "%" + "%d" % (self.versions[self.fname])
        self.versions[self.fname] += 1
        return name

    def output(self, ir_filename=None):
        '''
        outputs generated IR code to given file. If no file is given, output to stdout
        '''
        if ir_filename:
            print("Outputting IR to %s." % ir_filename)
            buf = open(ir_filename, 'w')
        else:
            print("Printing IR:\n\n")
            buf = sys.stdout
        for i,line in enumerate(self.code or []):
            pprint(line,buf)

    # You must implement visit_Nodename methods for all of the other
    # AST nodes.  In your code, you will need to make instructions
    # and append them to the self.code list.
    #
    # A few sample methods follow.  You may have to adjust depending
    # on the names of the AST nodes you've defined.

    def visit_Program(self, node):
        for i, child in enumerate(node.gdecls or []):
            self.visit(child)

    def visit_GlobalDecl(self, node):
        for i, child in enumerate(node.decls or []):
            self.visit(child)

    def visit_FuncDef(self, node):
        self.code.append(('define', '_{}'.format(node.decl.name.name)))

        self.visit(node.decl.type)
        self.visit(node.compound_statement)

    def visit_FuncDecl(self, node):
        # Check if function has parameters
        if node.args is not None:
            self.visit(node.args)

    def visit_FuncCall(self, node):
        for param in node.params.list:
            self.code.append(('param_' + getBasicType(param), param.name))

        self.code.append(('call', node.name.name))

    def visit_ParamList(self, node):
        for i, child in enumerate(node.list or []):
            self.visit(child)

    def visit_Cast(self, node):
        self.visit(node.expression)
        target = self.new_temp()
        node.temp_location = target
        if getBasicType(node.expression)=='int' and getBasicType(node)=='float':
            cast = 'sitofp'
        elif getBasicType(node.expression)=='float' and getBasicType(node)=='int':
            cast = 'fptosi'
        else:
            err = f"{node.coord.line}:{node.coord.column} - bad cast operation: should be from int to float or vice-versa only."
            assert False, err
        inst = (cast,node.expression.temp_location,target)
        self.code.append(inst)

    def visit_Compound(self, node):
        for i, child in enumerate(node.block_items or []):
            self.visit(child)

    def visit_Decl(self, node):
        self.visit(node.type)
        target = node.type.temp_location
        self.temp_var_dict[node.name.name] = target

        # Make the SSA opcode and append to list of generated instructions
        #inst = ('alloc_' + getBasicType(node.type), target)
        #self.code.append(inst)

        if node.init:
            self.visit(node.init)
            # target_store = self.new_temp(node.type.type)
            self.code.append(('store_' + getBasicType(node), node.init.temp_location, target))

    def visit_Return(self, node):
        target = self.new_temp()
        self.visit(node.expr)
        self.code.append(('store_' + getBasicType(node.expr), node.expr.temp_location, target))
        # Make the SSA opcode and append to list of generated instructions
        inst = ('return_' + getBasicType(node), target)
        self.code.append(inst)

    def visit_Constant(self, node):
        # Create a new temporary variable name
        #target = self.new_temp(node.type)
        target = self.new_temp()
        # Make the SSA opcode and append to list of generated instructions
        inst = ('literal_'+node.type.names[0], node.value, target)
        self.code.append(inst)

        # Save the name of the temporary variable where the value was placed
        node.temp_location = target

    def visit_ID(self, node):
        tmp = self.new_temp()
        node.temp_location = tmp
        source = self.temp_var_dict[node.name]
        tp = getBasicType(node)
        # load this variable in a new temp
        inst = ('load_'+tp,source,tmp)
        self.code.append(inst)

    def visit_BinaryOp(self, node):
        # Visit the left and right expressions
        self.visit(node.left)
        self.visit(node.right)
        # Make a new temporary for storing the result
        #target = self.new_temp(getInnerMostType(node.type))
        target = self.new_temp()
        # Create the opcode and append to list
        opcode = binary_ops[node.op]+"_"+getBasicType(node)
        inst = (opcode, node.left.temp_location, node.right.temp_location, target)
        self.code.append(inst)
        # Store location of the result on the node
        node.temp_location = target

        #inst = ('store_' + getBasicType(node.value), node.value.temp_location, self.temp_var_dict[node.assignee.name])
        #self.code.append(inst)

    def visit_ExprList(self, node):
        for i, child in enumerate(node.list or []):
            self.visit(child)
        node.temp_location = node.list[0].temp_location

    def visit_Print(self, node):
        # Visit the expression
        self.visit(node.expr)

        # Create the opcode and append to list
        inst = ('print_'+node.expr.type.name, node.expr.temp_location)
        self.code.append(inst)

    def visit_VarDecl(self, node):
        tp = getBasicType(node)
        tmp = self.new_temp()
        node.temp_location = tmp
        vid = '@'+node.name.name
        # if global store on heap 
        if node.isGlobal:
            inst = ('global_'+tp, vid)
        # otherwise, allocate on stack memory
        else:
            inst = ('alloc_'+tp, tmp)
        self.code.append(inst)

    def visit_LoadLocation(self, node):
        #target = self.new_temp(node.type)
        target = self.new_temp()
        inst = ('load_'+node.type.name,
                node.name,
                target)
        self.code.append(inst)
        node.temp_location = target

    def visit_Assignment(self, node):
        self.visit(node.value)
        tmp = self.temp_var_dict[node.assignee.name]
        self.temp_location = tmp
        t = getBasicType(node)
        inst = ('store_'+t, node.value.temp_location, tmp)
        self.code.append(inst)

    def visit_UnaryOp(self, node):
        self.visit(node.left)
        #target = self.new_temp(node.type)
        target = self.new_temp()
        opcode = unary_ops[node.op] + "_" + node.left.type.name
        inst = (opcode, node.left.temp_location)
        self.code.append(inst)
        node.temp_location = target

if __name__ == '__main__':

    import sys

    # open source code file and read contents
    filename = sys.argv[1]
    code = open(filename).read()
    # parse code and generate AST
    p = Parser()
    ast = p.parse(code)
    # perform semantic checks
    check = CheckProgramVisitor()
    check.visit_Program(ast)
    # generate IR
    gencode = GenerateCode()
    gencode.visit_Program(ast)
    ir_filename = filename[:-3] + '.code'
    gencode.output(ir_filename)
