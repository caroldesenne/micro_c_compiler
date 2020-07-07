from pprint import pprint
from parser import Parser
from ast import *
from checker import CheckProgramVisitor
from generator import GenerateCode
from interpreter import Interpreter
from graphviz import Digraph
from optimizer import *

from llvmlite import ir
from llvmlite import binding as llvmlite

from llvmlite import ir, binding
from ctypes import CFUNCTYPE, c_int

type_llvm_dict = {
    'int':      ir.IntType(32),
    'float':    ir.FloatType(),
    'bool':     ir.IntType(1),
    'char':     ir.IntType(8),
    'char_ptr': ir.IntType(8).as_pointer(),
    'string':   ir.IntType(8),
    'void':     ir.VoidType(),
}

def isLabel(instruction):
        op = instruction[0]
        # Not all instructions that have length 1 are labels. Other option is: ('return_void',), also ('print_void',)
        if (len(instruction) == 1) and (op not in ['return_void','print_void']):
            return op
        else:
            return None

def make_bytearray(buf):
    # Make a byte array constant from *buf*.
    b = bytearray(buf)
    n = len(b)
    return ir.Constant(ir.ArrayType(ir.IntType(8), n), b)

class LLVM_Converter(object):

    def __init__(self, cfg):
        self.cfg                = cfg
        self.module             = ir.Module()
        self.params             = [] # list to hold parameters to be passed to a function call
        self.builder            = None
        self.cur_func           = None
        self.temp_ptr_dict      = {}
        self.label_block_dict   = {}
        self.label_builder_dict = {}
        self.fname_fn_dict      = {}



        self.binding = binding
        self.binding.initialize()
        self.binding.initialize_native_target()
        self.binding.initialize_native_asmprinter()

        self.module = ir.Module(name=__file__)
        self.module.triple = self.binding.get_default_triple()

        self._create_execution_engine()

        # declare external functions
        self._declare_printf_function()
        self._declare_scanf_function()

    def get_ptr(self, label):
        if (self.cur_func, label) in self.temp_ptr_dict:
            return self.temp_ptr_dict[(self.cur_func, label)]
        if ('global', label) in self.temp_ptr_dict:
            return self.temp_ptr_dict[('global', label)]
        return None

    def set_ptr(self, label, ptr, is_global=False):
        if is_global:
            self.temp_ptr_dict[('global', label)] = ptr
        else:
            self.temp_ptr_dict[(self.cur_func, label)] = ptr

    def convert(self):
        for _, fcfg in self.cfg.func_cfg_dict.items():
            for label, block in fcfg.label_block_dict.items():
                for inst in block.instructions:
                    if 'define' in inst[0]:
                        self.cur_func = inst[1]

                        fn = self.convert_define(inst)
                        self.fname_fn_dict[self.cur_func] = fn

                        self.label_builder_dict[(self.cur_func, 'entry')] = self.builder

                    elif isLabel(inst):
                        # Workaround for adjacent labels
                        if len(block.instructions) == 1:
                            continue

                        self.label_block_dict[(self.cur_func, str(label))]   = fn.append_basic_block(str(label))
                        self.label_builder_dict[(self.cur_func, str(label))] = ir.IRBuilder(self.label_block_dict[(self.cur_func, str(label))])

        for _, fcfg in self.cfg.func_cfg_dict.items():
            for label, block in fcfg.label_block_dict.items():
                for inst in block.instructions:
                    if 'define' in inst[0]:
                        self.cur_func = inst[1]
                        self.builder  = self.label_builder_dict[(self.cur_func, 'entry')]
                    elif isLabel(inst):
                        if (self.cur_func, str(label)) in self.label_builder_dict:
                            self.builder = self.label_builder_dict[(self.cur_func, str(label))]
                    else:
                        op = inst[0]
                        op_without_type = op.split('_')[0]

                        method    = 'convert_' + op_without_type
                        converter = getattr(self, method, None)
                        if converter:
                            converter(inst)


        print('================================')
        print(self.module)
        print('================================')

        for fname, fn in self.fname_fn_dict.items():
            print(fname)
            # llvmlite.view_dot_graph(llvmlite.get_function_cfg(fn), view=True, filename=fname)
        print('================================')

    ####### Memory operations #######
    def convert_global(self, instruction):
        op      = instruction[0].split('_')
        op_type = op[1]
        target  = instruction[1][1:]
        value   = 0
        if len(instruction) > 2:
            value = instruction[2]

        # string
        if op_type == 'string':
            value = list(value)
            value.append('\00')
            array_type = ir.ArrayType(type_llvm_dict['char'], len(value))
            self.temp_ptr_dict[('global', target)] = llvmlite.ir.GlobalVariable(self.module, array_type, target)
            self.temp_ptr_dict[('global', target)].initializer = ir.Constant.literal_array([type_llvm_dict['char'](ord(v)) for v in list(value)])
        # array
        elif len(op)==3 or len(op)==4:
            size = 1
            if len(op) > 2:
                size *= int(op[2])
            if len(op) > 3:
                size *= int(op[3])
            array_type = ir.ArrayType(type_llvm_dict[op_type], size)
            self.temp_ptr_dict[('global', target)] = llvmlite.ir.GlobalVariable(self.module, array_type, target)
            if value:
                self.temp_ptr_dict[('global', target)].initializer = ir.Constant.literal_array([type_llvm_dict[op_type](v) for v in value])
            else:
                self.temp_ptr_dict[('global', target)].initializer = ir.Constant.literal_array([type_llvm_dict[op_type](0) for i in range(size)])
        else:
            self.temp_ptr_dict[('global', target)] = llvmlite.ir.GlobalVariable(self.module, type_llvm_dict[op_type], target)
            self.temp_ptr_dict[('global', target)].initializer = type_llvm_dict[op_type](value)
        self.temp_ptr_dict[('global', target)] = self.builder.bitcast(self.temp_ptr_dict[('global', target)], type_llvm_dict[op_type].as_pointer())


    def convert_elem(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]
        base    = instruction[1][1:]
        index   = instruction[2][1:]
        target  = instruction[3][1:]

        base_ptr   = self.get_ptr(base)
        zero_index = type_llvm_dict['int'](0)
        index_ptr  = self.get_ptr(index)
        self.temp_ptr_dict[(self.cur_func, target)] = self.builder.gep(base_ptr, [index_ptr])

    def convert_alloc(self, instruction):
        op      = instruction[0].split('_')
        op_type = op[1]
        name    = instruction[1][1:]
        size = 1
        if len(op) > 2:
            size = int(op[2])
        if len(op) > 3:
            size *= int(op[3])

        self.temp_ptr_dict[(self.cur_func, name)] = self.builder.alloca(type_llvm_dict[op_type], size=size, name=name)

    def convert_store(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]
        source  = instruction[1][1:]
        target  = instruction[2][1:]

        # TODO Store_*
        source_ptr = self.get_ptr(source)
        target_ptr = self.get_ptr(target)

        if len(op.split('_')) == 3 and op.split('_')[2] != '*':
            size = int(op.split('_')[2])

            if op_type == 'float':
                size *= 4
            elif op_type == 'int':
                size *= type_llvm_dict['int'].width//8

            if target_ptr:
                memcpy = self.module.declare_intrinsic('llvm.memcpy', [type_llvm_dict['char_ptr'], type_llvm_dict['char_ptr'], type_llvm_dict['int']])
                source_ptr = self.builder.bitcast(source_ptr, type_llvm_dict['char_ptr'])
                target_ptr = self.builder.bitcast(target_ptr, type_llvm_dict['char_ptr'])
                self.builder.call(memcpy, [target_ptr, source_ptr, type_llvm_dict['int'](size), type_llvm_dict['bool'](0)])
            else:
                self.temp_ptr_dict[(self.cur_func, target)] = source_ptr
        else:
            if target_ptr:
                if isinstance(source_ptr.type, llvmlite.ir.types.PointerType):
                    self.builder.store(self.builder.load(source_ptr), target_ptr)
                else:
                    self.builder.store(source_ptr, target_ptr)
            else:
                self.temp_ptr_dict[(self.cur_func, target)] = source_ptr

    def convert_load(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]
        source  = instruction[1][1:]
        target  = instruction[2][1:]

        source_ptr = self.get_ptr(source)
        if isinstance(source_ptr, ir.Constant):
            self.temp_ptr_dict[(self.cur_func, target)] = source_ptr
        else:
            self.temp_ptr_dict[(self.cur_func, target)] = self.builder.load(source_ptr)

    ####### Literal #######
    def convert_literal(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]
        target  = instruction[2][1:]
        literal = instruction[1]

        loc = self.get_ptr(target)
        const = type_llvm_dict[op_type](literal)
        if loc:
            self.builder.store(const, loc)
        else:
            self.temp_ptr_dict[(self.cur_func, target)] = const


    ####### Cast operations #######
    def convert_fptosi(self, instruction):
        source  = instruction[1][1:]
        target  = instruction[2][1:]

        source_ptr = self.get_ptr(source)
        self.temp_ptr_dict[(self.cur_func, target)] = self.builder.fptosi(source_ptr, type_llvm_dict['int'])

    def convert_sitofp(self, instruction):
        source  = instruction[1][1:]
        target  = instruction[2][1:]

        source_ptr = self.get_ptr(source)
        self.temp_ptr_dict[(self.cur_func, target)] = self.builder.sitofp(source_ptr, type_llvm_dict['float'])

    ####### Binary operations #######
    def convert_binary_op(self, instruction, func):
        left_op  = instruction[1][1:]
        right_op = instruction[2][1:]
        target   = instruction[3][1:]

        left = self.get_ptr(left_op)
        right = self.get_ptr(right_op)

        self.temp_ptr_dict[(self.cur_func, target)] = func(left, right)

    def convert_add(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]
        if op_type == 'float':
            func = self.builder.fadd
        else:
            func = self.builder.add
        self.convert_binary_op(instruction, func)

    def convert_sub(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]
        if op_type == 'float':
            func = self.builder.fsub
        else:
            func = self.builder.sub
        self.convert_binary_op(instruction, func)

    def convert_mul(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]
        if op_type == 'float':
            func = self.builder.fmul
        else:
            func = self.builder.mul
        self.convert_binary_op(instruction, func)

    def convert_div(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]
        if op_type == 'float':
            func = self.builder.fdiv
        else:
            func = self.builder.sdiv
        self.convert_binary_op(instruction, func)

    def convert_mod(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]
        if op_type == 'float':
            func = self.builder.frem
        else:
            func = self.builder.srem
        self.convert_binary_op(instruction, func)

    ####### Binary operations - boolean #######
    def convert_and(self, instruction):
        func = self.builder.and_
        self.convert_binary_op(instruction, func)

    def convert_xor(self, instruction):
        func = self.builder.xor
        self.convert_binary_op(instruction, func)

    def convert_or(self, instruction):
        func = self.builder.or_
        self.convert_binary_op(instruction, func)

    ####### Unary operations - boolean #######
    def convert_unary_op(self, instruction):
        op     = instruction[1][1:]
        target = instruction[2][1:]

        op = self.get_ptr(op)

        self.temp_ptr_dict[(self.cur_func, target)] = func(op)

    def convert_not(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]
        func = self.builder.not_
        self.convert_unary_op(instruction, func)

    def convert_compare(self, instruction, comp):
        left_op  = instruction[1][1:]
        right_op = instruction[2][1:]
        target   = instruction[3][1:]
        left = self.get_ptr(left_op)
        right = self.get_ptr(right_op)
        op      = instruction[0]
        op_type = op.split('_')[1]

        if op_type == 'float':
            comp = self.builder.fcmp_ordered(comp, left, right, name=target)
        else:
            comp = self.builder.icmp_signed(comp, left, right, name=target)
        self.temp_ptr_dict[(self.cur_func, target)] = comp

    def convert_lt(self, instruction):
        self.convert_compare(instruction, '<')

    def convert_le(self, instruction):
        self.convert_compare(instruction, '<=')

    def convert_ge(self, instruction):
        self.convert_compare(instruction, '>=')

    def convert_gt(self, instruction):
        self.convert_compare(instruction, '>')

    def convert_eq(self, instruction):
        self.convert_compare(instruction, '==')

    def convert_ne(self, instruction):
        self.convert_compare(instruction, '!=')


    ####### Print and Read #######
    def _global_constant(self, builder_or_module, name, value, linkage='internal'):
        # Get or create a (LLVM module-)global constant with *name* or *value*.
        if isinstance(builder_or_module, ir.Module):
            mod = builder_or_module
        else:
            mod = builder_or_module.module
        data = ir.GlobalVariable(mod, value.type, name=name)
        data.linkage = linkage
        data.global_constant = True
        data.initializer = value
        data.align = 1
        return data

    def _cio(self, fname, format, *target):
        # Make global constant for string format
        mod = self.builder.module
        fmt_bytes = make_bytearray((format + '\00').encode('ascii'))
        global_fmt = self._global_constant(mod, mod.get_unique_name('.fmt'), fmt_bytes)
        fn = mod.get_global(fname)
        ptr_fmt = self.builder.bitcast(global_fmt, ir.IntType(8).as_pointer())
        return self.builder.call(fn, [ptr_fmt] + list(target))

    def convert_print(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]

        if op_type == 'void':
            self._cio('printf', '\n')
        else:
            target   = instruction[1][1:]
            target_ptr = self.get_ptr(target)
            if op_type == 'int':
                self._cio('printf', '%d', target_ptr)
            elif op_type == 'float':
                self._cio('printf', '%.2f', target_ptr)
            elif op_type == 'char':
                self._cio('printf', '%c', target_ptr)
            elif op_type == 'string':
                self._cio('printf', '%s', target_ptr)

    def convert_read(self, instruction):
        op      = instruction[0]
        op_type = op.split('_')[1]
        target   = instruction[1][1:]

        target_ptr = self.get_ptr(target)
        if op_type == 'int':
            self._cio('scanf', '%d', target_ptr)
        elif op_type == 'float':
            self._cio('scanf', '%.2f', target_ptr)
        elif op_type == 'char':
            self._cio('scanf', '%c', target_ptr)
        elif op_type == 'string':
            self._cio('scanf', '%s', target_ptr)

    ####### Function operations #######
    def convert_param(self, instruction):
        source = instruction[1][1:]
        param  = self.get_ptr(source)
        self.params.append(param)

    def convert_call(self, instruction):
        fname  = instruction[1]
        target = instruction[2][1:]

        self.temp_ptr_dict[(self.cur_func, target)] = self.builder.call(self.fname_fn_dict[fname], self.params)
        self.params = [] # empty parameters list

    def convert_return(self, instruction):
        op      = instruction[0]
        if 'void' not in op:
            target  = instruction[1][1:]

            ret = self.get_ptr(target)
            self.builder.ret(ret)
        else:
            self.builder.ret_void()

    def convert_define(self, instruction):
        op              = instruction[0]
        op_without_type = op.split('_')[0]
        op_type         = op.split('_')[1]

        argTypes = [type_llvm_dict[arg[0]] for arg in instruction[2]]
        fnty     = ir.FunctionType(type_llvm_dict[op_type], argTypes)

        fn = ir.Function(self.module, fnty, instruction[1][1:])

        self.builder = ir.IRBuilder(fn.append_basic_block('entry'))

        for i,args in enumerate(instruction[2]):
            fn.args[i].name = args[1][1:]
            self.temp_ptr_dict[(self.cur_func, args[1][1:])] = fn.args[i]
        return fn


    ####### Block related operations #######
    def convert_cbranch(self, instruction):
        op           = instruction[0]
        cond         = instruction[1][1:]
        true_branch  = instruction[2][1:]
        false_branch = instruction[3][1:]

        cond_ptr  = self.temp_ptr_dict[(self.cur_func, cond)]
        true_ptr  = self.label_block_dict[(self.cur_func, true_branch)]
        false_ptr = self.label_block_dict[(self.cur_func, false_branch)]

        self.builder.cbranch(cond_ptr, true_ptr, false_ptr)

    def convert_jump(self, instruction):
        op     = instruction[0]
        target = instruction[1][1:]

        try:
            block = self.label_block_dict[(self.cur_func, target)]
            self.builder.branch(block)
        except:
            pass







    def _create_execution_engine(self):
        """
        Create an ExecutionEngine suitable for JIT code generation on
        the host CPU.  The engine is reusable for an arbitrary number of
        modules.
        """
        target = self.binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        # And an execution engine with an empty backing module
        backing_mod = binding.parse_assembly("")
        engine = binding.create_mcjit_compiler(backing_mod, target_machine)
        self.engine = engine

    def _declare_printf_function(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        printf = ir.Function(self.module, printf_ty, name="printf")
        self.printf = printf

    def _declare_scanf_function(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        scanf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        scanf = ir.Function(self.module, scanf_ty, name="scanf")
        self.scanf = scanf

    def _compile_ir(self):
        """
        Compile the LLVM IR string with the given engine.
        The compiled module object is returned.
        """
        # Create a LLVM module object from the IR
        # self.builder.ret_void()
        llvm_ir = str(self.module)
        mod = self.binding.parse_assembly(llvm_ir)
        mod.verify()
        # Now add the module and make sure it is ready for execution
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()
        return mod

    def save_ir(self, filename):
        with open(filename, 'w') as output_file:
            output_file.write(str(self.module))

    def execute_ir(self):
        mod = self._compile_ir()
        # Obtain a pointer to the compiled 'main' - it's the address of its JITed code in memory.
        main_ptr = self.engine.get_function_address('main')
        # To convert an address to an actual callable thing we have to use
        # CFUNCTYPE, and specify the arguments & return type.
        main_function = CFUNCTYPE(c_int)(main_ptr)
        # Now 'main_function' is an actual callable we can invoke
        res = main_function()
        print(res)

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
    pprint(gencode.code)
    # cfg.view()


    llvm = LLVM_Converter(cfg)
    llvm.convert()

    llvm.execute_ir()

    llvm.save_ir('llvm.ir')