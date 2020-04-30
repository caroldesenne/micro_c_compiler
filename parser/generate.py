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
    '<': 'lt',
    '<=': 'le',
    '>=': 'ge',
    '>': 'gt',
    '==': 'eq',
    '!=': 'ne',
    '&&': 'and',
    '||': 'or',
}

unary_ops = {
    '++': 'add_int',
    '--': 'sub_int',
    'p++': 'add_int',
    'p--': 'sub_int',
}

'''
TODO:

ArrayDecl *
ArrayRef *
InitList *
'''

class Labels(object):
    '''
    Class representing all the labels in a program (both local and global).
    Each scope level is represented by a symbol table, and they are assembled in a list.
    The first element of the list is the root scope (global variables) and we go into 
    deeper scopes as we go through the array. Depth represents the maximal scope depth we are in
    at the moment.
    '''
    def __init__(self):
        self.scope = []
        self.depth = 0
        self.constants = 0

    def pushLevel(self):
        s = SymbolTable()
        self.scope.append(s)
        self.depth += 1

    def popLevel(self):
        self.scope.pop()
        self.depth -= 1

    def createConstant(self):
        s = '@.const.'+str(self.constants)
        self.constants += 1
        return s

    def insertGlobal(self,sym,t):
        '''
        insert inserts a new symbol in the global symbol table (which is
        the one represented by self.scope[0]).
        '''
        s = self.scope[0]
        s.add(sym,t)

    def insert(self,sym,t):
        '''
        insert inserts a new symbol in the symbol table of the current scope (which is
        the one represented by depth, or self.scope[self.depth-1]).
        '''
        s = self.scope[self.depth-1]
        s.add(sym,t)

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
        self.globals = []
        self.code = []
        self.labels = Labels()

    def mergeCode(self):
        self.code = self.globals + self.code

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
        self.labels.pushLevel()
        for i, child in enumerate(node.gdecls or []):
            self.visit(child)
        self.labels.popLevel()
        self.mergeCode()

    def visit_GlobalDecl(self, node):
        for i, child in enumerate(node.decls or []):
            self.visit(child)

    def visit_FuncDef(self, node):
        self.visit(node.decl)

        for i, child in enumerate(node.decl_list or []):
            self.visit(child)

        self.visit(node.compound_statement)
        
        # insert exit label
        exit = self.labels.find("exit_func")
        self.code.append((exit[1:],))
        # insert return instruction
        bt = getBasicType(node)
        if bt=='void':
            self.code.append(('return_void',))
        else:
            rvalue = self.new_temp()
            ret = self.labels.find("return")
            self.code.append(('load_'+bt, ret, rvalue))
            self.code.append(('return_'+bt, rvalue))
        self.labels.popLevel()

    def visit_FuncDecl(self, node):
        if node.args:
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
        self.code.append((cast,node.expression.temp_location,target))

    def visit_Compound(self, node):
        for i, child in enumerate(node.block_items or []):
            self.visit(child)

    def visit_Decl(self, node):
        if isinstance(node.type, FuncDecl):
            if node.isFunction:
                self.code.append(('define', node.name.name))
                self.labels.pushLevel()
                self.visit(node.type)
                return_label = self.new_temp()
                exit_label = self.new_temp()
                self.labels.insert("return", return_label)
                self.labels.insert("exit_func", exit_label)
            else:
                pass # do nothing for prototype
        else: # can be ArrayDecl or VarDecl
            self.visit(node.type)
            if node.init:
                if(node.type.isGlobal): 
                # if global, we need to pop last appended code and insert init on it
                    line = self.globals.pop()
                    init = str(node.init.value)
                    self.globals.append((line[0],line[1],init))
                else:
                    self.visit(node.init)
                    target = node.type.temp_location
                    self.code.append(('store_' + getBasicType(node), node.init.temp_location, target))

    def visit_Return(self, node):
        bt = getBasicType(node)
        # store return value
        if bt != 'void':
            self.visit(node.expr)
            res = node.expr.temp_location
            ret = self.labels.find("return")
            self.code.append(('store_'+bt, res, ret))
        # jump to exit of function
        exit = self.labels.find("exit_func")
        self.code.append(('jump', exit))

    def visit_Assert(self, node):
        self.visit(node.expr)
        true_label = self.new_temp()
        false_label = self.new_temp()
        exit_label = self.new_temp()
        # branch between true and false assertion
        self.code.append(('cbranch', node.expr.temp_location, true_label, false_label))
        # true branch
        self.code.append((true_label[1:],))
        self.code.append(('jump', exit_label))
        # false branch
        self.code.append((false_label[1:],))
        # alloc the error string as a global variable
        const_name = self.labels.createConstant()
        fail = f"assertion_fail on {node.expr.coord.line}:{node.expr.coord.column}"
        self.globals.append(('global_string', const_name, fail))
        # refer to the global constant to print the error
        self.code.append(('print_string', const_name))
        exit = self.labels.find("exit_func")
        self.code.append(('jump', exit))
        self.code.append((exit_label[1:],))

    def visit_If(self, node):
        true_label = self.new_temp()
        false_label = self.new_temp()
        exit_label = self.new_temp()
        # test and branch
        self.visit(node.cond)
        self.code.append(('cbranch', node.cond.temp_location, true_label, false_label))
        # if statement
        self.code.append((true_label[1:],))
        self.visit(node.statement)
        # else statement exists
        if node.else_st:
            self.code.append(('jump', exit_label))
            self.code.append((false_label[1:],))
            self.visit(node.else_st)
            self.code.append((exit_label[1:],))
        else:
            self.code.append((false_label[1:],))

    def visit_While(self, node):
        entry_label = self.new_temp()
        body_label = self.new_temp()
        exit_label = self.new_temp()
        # TODO: do I need to insert this label as an exit_loop for the break to find? If yes, need to push new scope here?
        self.code.append((entry_label[1:],))
        # check condition
        self.visit(node.cond)
        self.code.append(('cbranch', node.cond.temp_location, body_label, exit_label))
        # start the while body
        self.code.append((body_label[1:],))
        # perform statement
        self.visit(node.statement)
        # jump back to beginning
        self.code.append(('jump', entry_label[1:]))
        # or end for
        self.code.append((exit_label[1:],))

    def visit_For(self, node):
        self.labels.pushLevel()
        entry_label = self.new_temp()
        body_label = self.new_temp()
        exit_label = self.new_temp()
        # record this for break
        self.labels.insert("exit_loop",exit_label)
        self.visit(node.init)
        self.code.append((entry_label[1:],))
        # check condition
        self.visit(node.stop_cond)
        self.code.append(('cbranch', node.stop_cond.temp_location, body_label, exit_label))
        # start the for body
        self.code.append((body_label[1:],))
        # perform statement and increment
        self.visit(node.statement)
        self.visit(node.increment)
        # jump back to beginning
        self.code.append(('jump', entry_label[1:]))
        # or end for
        self.code.append((exit_label[1:],))
        self.labels.popLevel()

    def visit_Break(self, node):
        target = self.labels.find("exit_loop")
        self.code.append(('jump', target))

    def visit_DeclList(self, node):
        for i, child in enumerate(node.decls or []):
            self.visit(child)

    def visit_Constant(self, node):
        # Create a new temporary variable name
        target = self.new_temp()
        # Make the SSA opcode and append to list of generated instructions
        self.code.append(('literal_'+node.type.names[0], node.value, target))
        # Save the name of the temporary variable where the value was placed
        node.temp_location = target

    def visit_ID(self, node):
        tmp = self.new_temp()
        source = self.labels.find(node.name)
        node.temp_location = tmp
        node.source = source
        tp = getBasicType(node)
        # load this variable in a new temp
        self.code.append(('load_'+tp,source,tmp))

    def visit_BinaryOp(self, node):
        # Visit the left and right expressions
        self.visit(node.left)
        self.visit(node.right)
        target = self.new_temp()
        # Binary operation opcode
        opcode = binary_ops[node.op]+"_"+getBasicType(node.left)
        self.code.append((opcode, node.left.temp_location, node.right.temp_location, target))
        # Store location of the result on the node
        node.temp_location = target

    def visit_ExprList(self, node):
        for i, child in enumerate(node.list or []):
            self.visit(child)
        node.temp_location = node.list[0].temp_location

    def visit_Print(self, node):
        if node.expr: # expression is not None
            self.visit(node.expr)
            inst = ('print_'+node.expr.type.name, node.expr.temp_location)
        else:
            inst = ('print_void',)
        self.code.append(inst)

    def visit_VarDecl(self, node):
        tp = getBasicType(node)
        vid = node.name.name
        # if global store on heap
        if node.isGlobal:
            self.labels.insertGlobal(vid,'@'+vid)
            self.globals.append(('global_'+tp, '@'+vid))
        # otherwise, allocate on stack memory
        else:
            tmp = self.new_temp()
            node.temp_location = tmp
            self.labels.insert(vid,tmp)
            self.code.append(('alloc_'+tp, tmp))

    def visit_Assignment(self, node):
        self.visit(node.value)
        # The assignee can be of two types: ID or ArrayRef
        if isinstance(node.assignee,ID):
            tmp = self.labels.find(node.assignee.name)
        elif isinstance(node.assignee,ArrayRef):
            self.visit(node.assignee)
            tmp = node.assignee.temp_location
        self.temp_location = tmp
        t = getBasicType(node)
        self.code.append(('store_'+t, node.value.temp_location, tmp))

    def visit_UnaryOp(self, node):
        self.visit(node.expression)
        temp_label = node.expression.temp_location
        target = self.new_temp()
        # perform the operation (add or sub 1)
        opcode = unary_ops[node.op]
        self.code.append((opcode, temp_label, 1, target))
        # store modified value back to original temp
        source = node.expression.source
        self.code.append(('store_int', target, source))
        # save this nodes temp location
        if node.op[0]=='p': # postifx_operation: save the initial temp_label value
            node.temp_location = temp_label
        else:
            node.temp_location = target

    def visit_Read(self, node):
        self.visit(node.expression)
        for exp in node.expression: # this is a list
            bt = getBasicType(exp)
            # read in a temp
            read_temp = self.new_temp()
            self.code.append(('read_'+bt, read_temp))
            # and store the value read in the exp location
            self.code.append(('store_'+getBasicType(node.expression), read_temp, exp.temp_location))

    def visit_Type(self, node):
        pass

    def visit_EmptyStatement(self, node):
        pass

if __name__ == '__main__':
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
