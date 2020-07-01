import sys
import ply.yacc as yacc
#import uctype
from pprint import pprint
from parser import Parser
from enum import Enum
from collections import defaultdict
from ast import *
from checker import *

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

assignments = {
    '+=': 'add',
    '-=': 'sub',
    '*=': 'mul',
    '/=': 'div',
    '%=': 'mod',
}

unary_ops = {
    '++': 'add_int',
    '--': 'sub_int',
    'p++': 'add_int',
    'p--': 'sub_int',
    '-' : 'sub_int', 
    '+' : 'whatever',
    '!' : 'not_bool',
}

class Phase(Enum):
    START = 0
    ALLOC = 1
    INIT = 2
    STATEMENT = 3

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
        self.loopStack = []
        self.constants = 0

    def insertExitLoop(self,label):
        self.loopStack.append(label)

    def popExitLoop(self):
        self.loopStack.pop()

    def getExitLoop(self):
        return self.loopStack[-1]

    def pushLevel(self):
        s = SymbolTable()
        self.scope.append(s)
        self.depth += 1

    def popLevel(self):
        self.scope.pop()
        self.depth -= 1

    def createConstant(self):
        s = '@.str.'+str(self.constants)
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
        self.versions = {self.fname:1}
        # The generated code (list of tuples)
        self.globals = []
        self.code = []
        self.labels = Labels()
        self.phase = Phase.START

    def mergeCode(self):
        self.code = self.globals + self.code

    def new_temp(self):
        '''
        Create a new temporary variable of a given scope (function name).
        '''
        if self.fname not in self.versions:
            self.versions[self.fname] = 1
        name = "%" + "%d" % (self.versions[self.fname])
        self.versions[self.fname] += 1
        return name

    def loadExpression(self,exp):
        '''
        performs a load operation and returns the temporary
        of the resulting load.
        '''
        tmp = self.new_temp()
        bt = getBasicType(exp)
        inst = 'load_'+bt
        if isinstance(exp, ArrayRef):
            inst += '_*'
        self.code.append((inst,exp.temp_location,tmp))
        return tmp

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
        # add \n for printing strings, which is always:
        # commenting this to match susy output
        # ('global_string', '@.str.0', '\n')
        # self.globals.append(('global_string',self.labels.createConstant(),'\n'))
        for i, child in enumerate(node.gdecls or []):
            self.visit(child)
        self.labels.popLevel()
        self.mergeCode()

    def visit_GlobalDecl(self, node):
        for i, child in enumerate(node.decls or []):
            self.visit(child)

    #############################################################################
    # From here till FuncDef, all these functions are used for FuncDef purposes #
    #############################################################################
    def FuncDefStart(self,node):
        self.phase = Phase.START
        self.visit(node.decl)
        # reserve temps for the parameters of the function
        if node.param_list:
            for p in node.param_list.list:
                aux = self.new_temp()
        # reserve temp for return
        return_label = self.new_temp()
        tp = getBasicType(node)
        if tp!='void':
            self.code.append(('alloc_'+tp, return_label))
        self.labels.insert("return", return_label)

    def FuncDefAlloc(self,node):
        # Start allocation phase
        self.phase = Phase.ALLOC
        # alloc temps for arguments
        self.visit(node.decl)
        # reserve temp for exit label
        exit_label = self.new_temp()
        self.labels.insert("exit_func", exit_label)
        # Go through all declarations
        for i,d in enumerate(node.decl_list or []):
            self.visit(d)

    def FuncDefStoreParams(self,node):
        if node.param_list:
            i = 1
            j = len(node.param_list.list)+2
            for p in node.param_list.list:
                bt = getBasicType(p)
                self.code.append(('store_'+bt,'%'+str(i),'%'+str(j)))
                i += 1
                j += 1

    def FuncDefInit(self,node):
        self.phase = Phase.INIT
        for i,d in enumerate(node.decl_list or []):
            self.visit(d)

    def FuncDefStatement(self,node):
        # visit the statements
        self.phase = Phase.STATEMENT
        if node.compound_statement:
            self.visit(node.compound_statement)

    def visit_FuncDef(self,node):
        # start new function with new temps
        self.fname = node.decl.name.name
        #####################################
        self.FuncDefStart(node)
        self.FuncDefAlloc(node)
        self.FuncDefStoreParams(node)
        # self.FuncDefInit(node)
        self.FuncDefStatement(node)
        ####################################
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
    #######################################################################
    #            Here end the functions used for FuncDef purposes         #
    #######################################################################

    def visit_FuncDecl(self, node):
        if node.args:
            self.visit(node.args)

    def visit_FuncCall(self, node):
        bt = getBasicType(node)
        # load params
        all_params = []
        for param in node.params.list:
            self.visit(param)
            tmp = param.temp_location
            if isinstance(param,ID) or isinstance(param, ArrayRef):
                tmp = self.loadExpression(param)
            all_params.append(('param_' + getBasicType(param), tmp))
        # put all params together
        for p in all_params:
            self.code.append(p)
        # call function
        result = self.new_temp()
        self.code.append(('call_'+bt, '@'+node.name.name, result))
        # store result label from call
        node.temp_location = result

    def visit_ParamList(self, node):
        for i, child in enumerate(node.list or []):
            self.visit(child)

    def visit_Cast(self, node):
        self.visit(node.expression)
        tmp = node.expression.temp_location
        # take expression out of list
        if isinstance(node.expression, ExprList):
            node.expression = node.expression.list[0]
        if isinstance(node.expression,ID) or isinstance(node.expression, ArrayRef):
            tmp = self.loadExpression(node.expression)
        target = self.new_temp()
        node.temp_location = target
        if getBasicType(node.expression)=='int' and getBasicType(node)=='float':
            cast = 'sitofp'
        elif getBasicType(node.expression)=='float' and getBasicType(node)=='int':
            cast = 'fptosi'
        else:
            err = f"{node.coord.line}:{node.coord.column} - bad cast operation: should be from int to float or vice-versa only."
            assert False, err
        self.code.append((cast,tmp,target))

    def visit_Compound(self, node):
        for i, child in enumerate(node.block_items or []):
            self.visit(child)

    def visit_InitList(self,node):
        arr = []
        for e in node.inits:
            if isinstance(e,Constant):
                arr.append(e.value)
            else:
                e.baseArray = False
                self.visit(e)
                arr.append(e.code)
        node.code = arr
        if node.baseArray:
            const_name = self.labels.createConstant()
            bt = getBasicType(node)
            sizes = self.buildArraySize(node.sizes)
            self.globals.append(('global_'+bt+sizes,const_name,arr))
            node.temp_location = const_name

    def buildArraySize(self,list):
        s = ''
        acc = 1
        for c in reversed(list):
            acc *= c.value
            s = '_'+str(acc)+s
        return s

    def visit_ArrayDecl(self,node):
        # build sizes string
        size = self.buildArraySize(node.size)
        tp = getBasicType(node)
        vid = getArrayName(node)
        # if global store on heap
        if node.isGlobal:
            self.labels.insertGlobal(vid,'@'+vid)
            self.globals.append(('global_'+tp+size, '@'+vid))
        # otherwise, allocate on stack memory
        else:
            tmp = self.new_temp()
            node.temp_location = tmp
            self.labels.insert(vid,tmp)
            self.code.append(('alloc_'+tp+size, tmp))

    def visit_ArrayRef2D(self,node):
        # v[i][j]
        decl = node.name.name.type
        # get size of inner array
        self.visit(decl.size[1])
        inSizeLabel = decl.size[1].temp_location

        # get access value label i
        i = node.name.access_value
        self.visit(i)
        if isinstance(i, ExprList):
            i = i.list[0]
        # load inner access value i
        acc1 = i.temp_location
        if isinstance(i, ID) or isinstance(i, ArrayRef):
            acc1 = self.loadExpression(i)

        # get to v[i]
        mul_target = self.new_temp()
        self.code.append(('mul_int',inSizeLabel,acc1,mul_target))

        # get access value label j
        j = node.access_value
        self.visit(j)
        if isinstance(j, ExprList):
            j = j.list[0]
        # load outer access value j
        acc2 = j.temp_location
        if isinstance(j, ID) or isinstance(j, ArrayRef):
            acc2 = self.loadExpression(j)

        # get to v[i][j]
        add_target = self.new_temp()
        self.code.append(('add_int',mul_target,acc2,add_target))
        
        # get base array label
        base_array = self.getBaseArray(decl)
        target = self.new_temp()
        bt = getBasicType(node)
        self.code.append(('elem_'+bt,base_array,add_target,target))
        node.temp_location = target

    def getArrayName(self,node):
        t = node
        while not isinstance(t,VarDecl):
            t = t.type
        return t.name.name

    def getBaseArray(self,node):
        if node.isGlobal:
            global_name = getArrayName(node)
            base_array = '@'+global_name
        else:
            base_array = node.temp_location
        return base_array

    def visit_ArrayRef(self,node):
        # 2D array reference
        if isinstance(node.name, ArrayRef):
            self.visit_ArrayRef2D(node)
        # 1D array reference
        else:
            self.visit(node.access_value)
            if isinstance(node.access_value, ExprList):
                node.access_value = node.access_value.list[0]
            # get access value label
            acc = node.access_value.temp_location
            if isinstance(node.access_value, ID) or isinstance(node.access_value, ArrayRef):
                acc = self.loadExpression(node.access_value)
            # get base array label
            base_array = self.getBaseArray(node.name.type)
            bt = getBasicType(node)
            target = self.new_temp()
            self.code.append(('elem_'+bt,base_array,acc,target))
            node.temp_location = target

    #######################################################################
    # From here till Decl, all these functions are used for Decl purposes #
    #######################################################################
    def visit_DeclFuncDecl(self,node):
        if node.isPrototype:
            pass # do nothing for prototype
        else: # this is a function definition    
            if self.phase == Phase.START:
                inst = 'define_' + getBasicType(node)
                if node.type.args is not None:
                    arguments = [(getBasicType(arg),'%'+str(i+1)) for i,arg in enumerate(node.type.args.list)]
                else:
                    arguments = []
                self.code.append((inst, '@'+node.name.name, arguments))
                # return_label = self.new_temp()
                # exit_label = self.new_temp()
                self.labels.pushLevel()
                # self.labels.insert("return", return_label)
                # self.labels.insert("exit_func", exit_label)
            else:
                self.visit(node.type)

    def visit_Decl(self, node):
        if isinstance(node.type, FuncDecl):
            self.visit_DeclFuncDecl(node)
        else: # can be ArrayDecl or VarDecl
            if node.type.isGlobal:
            # if global, we need to pop last appended code and insert init on it
                if isinstance(node.type, ArrayDecl):
                    self.visit(node.type)
                    if node.init:
                        line = self.globals.pop()
                        self.visit(node.init)
                        aux = self.globals.pop()
                        init = aux[2]
                        self.globals.append((line[0],line[1],init))
                else:
                    self.visit(node.type)
                    line = self.globals.pop()
                    if node.init:
                        init = node.init.value
                        self.globals.append((line[0],line[1],init))
            else:
                if self.phase==Phase.ALLOC:
                    self.visit(node.type)
                elif node.init:
                    self.visit(node.init)
                    target = node.type.temp_location
                    source = node.init.temp_location
                    if isinstance(node.init,ID) or isinstance(node.init, ArrayRef):
                        source = self.loadExpression(node.init)
                    inst = 'store_'+getBasicType(node)
                    if isinstance(node.type, ArrayDecl):
                        inst += self.buildArraySize(node.type.size)
                    self.code.append((inst, source, target))
    #######################################################################
    #             Here ends the functions used for Decl purposes          #
    #######################################################################

    def visit_Return(self, node):
        bt = getBasicType(node)
        # store return value
        if bt != 'void':
            self.visit(node.expr)
            # get return value from ExprList (it is always the first and unique element)
            node.expr = node.expr.list[0]
            res = node.expr.temp_location
            if isinstance(node.expr,ID) or isinstance(node.expr, ArrayRef):
                res = self.loadExpression(node.expr)
            ret = self.labels.find("return")
            # store result
            inst = 'store_'+bt
            self.code.append((inst, res, ret))
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
            self.code.append(('jump', exit_label))
            self.code.append((exit_label[1:],))
        else:
            self.code.append(('jump', false_label))
            self.code.append((false_label[1:],))

    def visit_While(self, node):
        entry_label = self.new_temp()
        body_label = self.new_temp()
        exit_label = self.new_temp()
        self.code.append(('jump', entry_label))
        # record this label for break
        self.labels.insertExitLoop(exit_label)
        # start of the while
        self.code.append((entry_label[1:],))
        # check condition
        self.visit(node.cond)
        self.code.append(('cbranch', node.cond.temp_location, body_label, exit_label))
        # start the while body
        self.code.append((body_label[1:],))
        # perform statement
        self.visit(node.statement)
        # jump back to beginning
        self.code.append(('jump', entry_label))
        # or end for
        self.code.append((exit_label[1:],))
        self.labels.popExitLoop()

    def visit_For(self, node):
        self.labels.pushLevel()
        entry_label = self.new_temp()
        body_label = self.new_temp()
        exit_label = self.new_temp()
        # record this for break
        self.labels.insertExitLoop(exit_label)
        # start of the for loop
        self.visit(node.init)
        self.code.append(('jump',entry_label))
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
        self.code.append(('jump', entry_label))
        # or end for
        self.code.append((exit_label[1:],))
        self.labels.popExitLoop()
        self.labels.popLevel()

    def visit_Break(self, node):
        target = self.labels.getExitLoop()
        self.code.append(('jump', target))

    def visit_DeclList(self, node):
        for i, child in enumerate(node.decls or []):
            self.visit(child)

    def visit_Constant(self, node):
        # if it is a string, save as global var
        if getBasicType(node)=='string':
            # alloc the error string as a global variable
            const_name = self.labels.createConstant()
            self.globals.append(('global_string', const_name, node.value[1:-1]))
            node.temp_location = const_name
        else:
            # Create a new temporary variable name
            target = self.new_temp()
            # Make the SSA opcode and append to list of generated instructions
            self.code.append(('literal_'+node.type.names[0], node.value, target))
            # Save the name of the temporary variable where the value was placed
            node.temp_location = target

    def visit_ID(self, node):
        source = self.labels.find(node.name)
        node.temp_location = source

    def visit_BinaryOp(self, node):
        # Visit the left and right expressions
        self.visit(node.left)
        self.visit(node.right)
        # load both left and right
        tmp1 = node.left.temp_location
        tmp2 = node.right.temp_location
        # take nodes out of lists
        if isinstance(node.left, ExprList):
            node.left = node.left.list[0]
        if isinstance(node.right, ExprList):
            node.right = node.right.list[0]
        # load if ID
        if isinstance(node.left,ID) or isinstance(node.left, ArrayRef):
            tmp1 = self.loadExpression(node.left)
        if isinstance(node.right,ID) or isinstance(node.right, ArrayRef):
            tmp2 = self.loadExpression(node.right)
        bt = getBasicType(node.left)
        # perform the binary operation
        target = self.new_temp()
        opcode = binary_ops[node.op]+"_"+getBasicType(node.left)
        self.code.append((opcode, tmp1, tmp2, target))
        # Store location of the result on the node
        node.temp_location = target

    def visit_ExprList(self, node):
        for i, child in enumerate(node.list or []):
            self.visit(child)
        node.temp_location = node.list[0].temp_location

    def visit_Print(self, node):
        if node.expr: # expression is not None
            for exp in node.expr.list:
                self.visit(exp)
                tmp = exp.temp_location
                if isinstance(exp,ID) or isinstance(exp, ArrayRef):
                    tmp = self.loadExpression(exp)
                bt = getBasicType(exp)
                self.code.append(('print_'+bt, tmp))
        else:
            self.code.append(('print_void',))
        # print an empty line after every print
        # commenting this to match Susy output
        # self.code.append(('print_string', '@.str.0'))

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
        bt = getBasicType(node)
        # load value
        self.visit(node.value)
        val = node.value.temp_location
        # load ID
        if isinstance(node.value, ID) or isinstance(node.value, ArrayRef):
            val = self.loadExpression(node.value)
        # The assignee can be of two types: ID or ArrayRef
        self.visit(node.assignee)
        if node.op =='=':
            tmp = val
        else: # not a simple assignment, so need to perform operations
            aux = self.loadExpression(node.assignee)
            tmp = self.new_temp()
            self.code.append((assignments[node.op]+'_'+bt,aux,val,tmp))
        origin = node.assignee.temp_location
        self.temp_location = origin
        inst = 'store_'+bt
        if isinstance(node.assignee, ArrayRef):
            inst += '_*'
        self.code.append((inst, tmp, origin))

    #############################################################################
    # From here till UnaryOp, all these functions are used for UnaryOp purposes #
    #############################################################################
    def plusPlusMinusMinus(self,node,source):
        # perform the operation (add or sub 1)
        opcode = unary_ops[node.op]
        # load 1 in a new temp
        one = self.new_temp()
        self.code.append(('literal_int', 1, one))
        target = self.new_temp()
        self.code.append((opcode, source, one, target))
        # store modified value back to original temp
        bt = getBasicType(node)
        inst = 'store_'+bt
        if isinstance(node.expression, ArrayRef):
            inst += '_*'
        self.code.append((inst, target, node.expression.temp_location))
        # save this nodes temp location
        if node.op[0]=='p': # postifx_operation: save the initial source value
            node.temp_location = source
        else:
            node.temp_location = target

    def minus(self,node,source):
        bt = getBasicType(node)
        # take the negative value of x: -x = 0-x
        opcode = unary_ops[node.op]
        # load 0 in a new temp
        zero = self.new_temp()
        self.code.append(('literal_'+bt, 0, zero))
        target = self.new_temp()
        self.code.append((opcode, zero, source, target))
        node.temp_location = target

    def notBool(self,node,source):
        opcode = unary_ops[node.op]
        target = self.new_temp()
        self.code.append((opcode, source, target))
        node.temp_location = target

    def visit_UnaryOp(self, node):
        self.visit(node.expression)
        # load whatever was there
        source = node.expression.temp_location
        # take expression out of list
        if isinstance(node.expression, ExprList):
            node.expression = node.expression.list[0]
        if isinstance(node.expression,ID) or isinstance(node.expression, ArrayRef):
            source = self.loadExpression(node.expression)
        # perform the unary operation
        if node.op=='+':
            node.temp_location = source
        elif node.op=='-':
            self.minus(node,source)
        elif node.op=='!':
            self.notBool(node,source)
        else: # ++, --, p++, p--
            self.plusPlusMinusMinus(node,source)
    #######################################################################
    #            Here end the functions used for UnaryOp purposes         #
    #######################################################################

    def visit_Read(self, node):
        for exp in node.expr.list:
            self.visit(exp)
            bt = getBasicType(exp)
            # read in the exp location
            inst = 'read_'+bt
            if isinstance(exp, ArrayRef):
                inst += '_*'
            self.code.append((inst, exp.temp_location))

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
