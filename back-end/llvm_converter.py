from pprint import pprint
from parser import Parser
from ast import *
from checker import CheckProgramVisitor
from generator import GenerateCode
from interpreter import Interpreter
from graphviz import Digraph
from optimizer import *

from llvmlite import ir
from llvmlite import binding as llvm

type_llvm_dict = {
    'int': ir.IntType(32),
    'float': ir.FloatType(),
    'bool': ir.IntType(8),
    'char': ir.IntType(8),
    'void': ir.VoidType(),
}

def isLabel(instruction):
        op = instruction[0]
        # Not all instructions that have length 1 are labels. Other option is: ('return_void',), also ('print_void',)
        if (len(instruction) == 1) and (op not in ['return_void','print_void']):
            return op
        else:
            return None

class LLVM_Converter(object):

    def __init__(self, cfg):
        self.cfg     = cfg
        self.module  = ir.Module()
        self.builder = None
        self.temp_ptr_dict = {}


    def convert(self):
        for _, fcfg in self.cfg.func_cfg_dict.items():
            for label, block in fcfg.label_block_dict.items():
                for inst in block.instructions:
                    if 'define' in inst[0]:
                        print(inst)
                        fn = self.convert_define(inst)

                        builder = ir.IRBuilder(fn.append_basic_block('entry'))
                        if self.builder == None:
                            self.builder = builder

                        # dot = llvm.get_function_cfg(fn)
                        # llvm.view_dot_graph(dot)
                    elif isLabel(inst):
                    #     lab = isLabel(inst)
                        pass
                    else:
                        op = inst[0]
                        op_without_type = op.split('_')[0]

                        method    = 'convert_' + op_without_type
                        converter = getattr(self, method, None)
                        print(converter)
                        if converter:
                            converter(inst)
        print('================================')
        pprint(self.module)



    def convert_alloc(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]
        name    = instruction[1][1:]
        alloc   = self.builder.alloca(type_llvm_dict[op_type], name=name)
        self.temp_ptr_dict[name] = alloc

    def convert_literal(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]
        target  = instruction[2][1:]
        literal = instruction[1]

        self.temp_ptr_dict[target] = self.builder.alloca(type_llvm_dict[op_type], name=target)
        self.builder.store_reg(ir.Constant(type_llvm_dict[op_type], literal), type_llvm_dict[op_type], target)

        print(self.temp_ptr_dict)

    def convert_store(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]
        source  = instruction[1][1:]
        target  = instruction[2][1:]

        #TODO Store_*
        if target not in self.temp_ptr_dict:
            self.temp_ptr_dict[target] = self.builder.alloca(type_llvm_dict[op_type], name=target)
        # self.builder.store_reg(self.builder.load(self.temp_ptr_dict[source]), type_llvm_dict[op_type], target)
        alloc = self.builder.store_reg(self.builder.load(self.temp_ptr_dict[source]), type_llvm_dict[op_type], target)
        # TODO should we do something with the alloc?

    def convert_load(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]
        source  = instruction[1][1:]
        target  = instruction[2][1:]

        #TODO Load_*
        if target not in self.temp_ptr_dict:
            self.temp_ptr_dict[target] = self.builder.alloca(type_llvm_dict[op_type], name=target)
        # self.builder.store_reg(self.builder.load(self.temp_ptr_dict[source]), type_llvm_dict[op_type], target)
        alloc = self.builder.store_reg(self.builder.load(self.temp_ptr_dict[source]), type_llvm_dict[op_type], target)
        # TODO should we do something with the alloc?

    def convert_branch_condition(self, instruction, comp):
        op      = instruction[0]
        op_type = op.split('_')[1]
        op1     = instruction[1][1:]
        op2     = instruction[2][1:]
        target  = instruction[3][1:]

        pred = self.builder.icmp_signed(comp, self.temp_ptr_dict[op1], self.temp_ptr_dict[op2], name=target)
        print(pred)

    def convert_lt(self, instruction):
        self.convert_branch_condition(instruction, '<')

    def convert_define(self, instruction):
        op = instruction[0]
        op_without_type = op.split('_')[0]
        op_type = op.split('_')[1]

        argTypes = [type_llvm_dict[arg[0]] for arg in instruction[2]]
        fnty = ir.FunctionType(type_llvm_dict[op_type], argTypes)

        fn = ir.Function(self.module, fnty, instruction[1][1:])
        for i,args in enumerate(instruction[2]):
            fn.args[i].name = args[1][1:]
        return fn

if __name__ == "__main__":
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

    cfg = CFG_Program(gencode.code)

    # cfg.view()


    llvm = LLVM_Converter(cfg)
    llvm.convert()

