import sys
from pprint import pprint
from parser import Parser
from ast import *
from checker import CheckProgramVisitor
from generator import GenerateCode
from interpreter import Interpreter

op_lambdas = {
    'add': lambda l, r: l + r,
    'sub': lambda l, r: l - r,
    'mul': lambda l, r: l * r,
    'div': lambda l, r: l // r,
    'mod': lambda l, r: l % r,
    'and': lambda l, r: l & r,
    'or':  lambda l, r: l | r,
    #'not': lambda l, r: l  r,
    'ne':  lambda l, r: int(l != r),
    'eq':  lambda l, r: int(l == r),
    'lt':  lambda l, r: int(l < r),
    'le':  lambda l, r: int(l <= r),
    'gt':  lambda l, r: int(l > r),
    'ge':  lambda l, r: int(l >= r),
}

class Block(object):
    def __init__(self, label):
        self.instructions = []   # Instructions in the block
        self.next_block   = None # Link to the next block
        self.parents      = []
        self.label        = label

        # Reaching definition (rd) stuff
        self.rd_gen  = set()
        self.rd_kill = set()
        self.rd_in   = set()
        self.rd_out  = set()

        # Liveness stuff
        self.live_gen  = set()
        self.live_kill = set()
        self.live_in   = set()
        self.live_out  = set()

    def append(self,instr):
        self.instructions.append(instr)

    def add_parent(self, parent_block):
        self.parents.append(parent_block)

    def __iter__(self):
        return iter(self.instructions)

class BasicBlock(Block):
    pass

class ConditionalBlock(Block):
    def __init__(self, label):
        super(ConditionalBlock, self).__init__(label)
        self.true_branch  = None
        self.false_branch = None

class CFG():
    def __init__(self, gen_code):
        self.gen_code             = gen_code
        self.label_block_dict     = {}
        self.label_blocktype_dict = {}
        self.first_block          = None

        self.create_blocks()

    '''
    Get type from instruction (basically if label or not) and return label converted to int.
    '''
    def get_instruction_type(self, instruction):
        op = instruction[0]
        # Not all instructions that have length 1 are labels. Other option is: ('return_void',)
        if (len(instruction) == 1) and (op != 'return_void'):
            return int(op)
        else:
            return op

    '''
    Creates a new block with a given block type and label
    '''
    def new_block(self, blocktype, label):
        if blocktype == 'jump' or blocktype == 'basic':
            return BasicBlock(label)
        elif blocktype == 'cbranch':
            return ConditionalBlock(label)

    def create_block_if_inexistent(self, label):
        if label not in self.label_block_dict.keys():
            self.label_block_dict[label] = self.new_block(self.label_blocktype_dict[label], label)

    def output(self):
        '''
        outputs generated IR code to given file. If no file is given, output to stdout
        '''
        for k, v in self.label_block_dict.items():
            print()
            print("Block {}".format(k))
            for line,code in enumerate(v.instructions):
                print("    "+str(line)+": "+str(code))
            if isinstance(v, BasicBlock) and v.next_block:
                print("Next block {}".format(v.next_block.label))

    def output_optimized_code(self):
        for k, v in self.label_block_dict.items():
            for line,code in enumerate(v.instructions):
                print(code)

    def output_rd(self):
        '''
        outputs generated IR code to given file. If no file is given, output to stdout
        '''
        print('=========================== RD: GEN and KILL ===========================')
        for label, block in self.label_block_dict.items():
            print()
            print('Block ', label)
            print('    GEN : ', sorted(list(block.rd_gen), key=lambda x: (x[0], x[1])))
            print('    KILL: ', sorted(list(block.rd_kill), key=lambda x: (x[0], x[1])))

        print('============================ RD: IN and OUT ============================')
        for label, block in self.label_block_dict.items():
            print()
            print('Block ', label)
            print('    IN : ', sorted(list(block.rd_in), key=lambda x: (x[0], x[1])))
            print('    OUT: ', sorted(list(block.rd_out), key=lambda x: (x[0], x[1])))

    def output_liveness(self):
        '''
        outputs generated IR code to given file. If no file is given, output to stdout
        '''
        print('=========================== LIVENESS: GEN and KILL ===========================')
        for label, block in self.label_block_dict.items():
            print()
            print('Block ', label)
            print('    GEN : ', sorted(list(block.live_gen),  key=lambda x: int(x[1:]) if x[0] == '%' else 0))
            print('    KILL: ', sorted(list(block.live_kill), key=lambda x: int(x[1:]) if x[0] == '%' else 0))

        print('============================ LIVENESS: IN and OUT ============================')
        for label, block in self.label_block_dict.items():
            print()
            print('Block ', label)
            print('    IN : ', sorted(list(block.live_in),  key=lambda x: int(x[1:]) if x[0] == '%' else 0))
            print('    OUT: ', sorted(list(block.live_out), key=lambda x: int(x[1:]) if x[0] == '%' else 0))

    def print(self):
        for k, v in self.label_block_dict.items():
            print('-------------------- Block {}:  --------------------'.format(k))
            for code in v.instructions:
                print(code)
            try:
                print('------ Parents -------')
                for parent in v.parents:
                    print(parent.label)
                if isinstance(v, ConditionalBlock):
                    print('---- true branch label', v.true_branch.label, 'false branch label', v.false_branch.label, '----')
                elif isinstance(v, BasicBlock):
                    print('--------------------- jump', v.next_block.label, '----------------------')
            except:
                pass

    '''
    Create CFG in two steps. First find all labels and
    determine the type of each block (basic or conditional).
    Then, creates the blocks, populates the instructions array
    and links them accordingly.
    '''
    def create_blocks(self):
        ####################################################################
        # First iteration, creating a mapping between block label and type #
        ####################################################################
        # Dummy label at the end
        self.gen_code.append(('0',))

        cur_label = 0
        for index in range(len(self.gen_code)):
            op = self.get_instruction_type(self.gen_code[index])
            if isinstance(op, int):
                prev_op = self.get_instruction_type(self.gen_code[index-1])
                if prev_op == 'jump' or prev_op == 'cbranch':
                    self.label_blocktype_dict[cur_label] = prev_op
                else:
                    self.label_blocktype_dict[cur_label] = 'basic'
                cur_label = op

        # remove  dummy label at the end
        del self.gen_code[-1]


        ##########################################################
        # Second iteration, creating the blocks and linking them #
        ##########################################################

        # Create first block
        self.first_block = self.new_block(self.label_blocktype_dict[0], 0)
        self.label_block_dict[0] = self.first_block

        cur_block = self.first_block
        for index in range(len(self.gen_code)):
            op = self.get_instruction_type(self.gen_code[index])
            # if we find a label, it means we are done with previous block
            if isinstance(op, int):
                # creates next block
                next_label = op
                self.create_block_if_inexistent(next_label)

                # link current block (next/jump/branch)
                if self.label_blocktype_dict[cur_block.label] == 'basic':
                    cur_block.next_block = self.label_block_dict[next_label]
                    self.label_block_dict[next_label].add_parent(cur_block)

                elif self.label_blocktype_dict[cur_block.label] == 'jump':
                    # dirty way to get jump label from instruction
                    jump_label = int(self.gen_code[index-1][1][1:])
                    self.create_block_if_inexistent(jump_label)
                    cur_block.next_block = self.label_block_dict[jump_label]
                    self.label_block_dict[jump_label].add_parent(cur_block)

                elif self.label_blocktype_dict[cur_block.label] == 'cbranch':
                    true_label  = int(self.gen_code[index-1][2][1:])
                    self.create_block_if_inexistent(true_label)
                    cur_block.true_branch  = self.label_block_dict[true_label]
                    self.label_block_dict[true_label].add_parent(cur_block)

                    false_label = int(self.gen_code[index-1][3][1:])
                    self.create_block_if_inexistent(false_label)
                    cur_block.false_branch = self.label_block_dict[false_label]
                    self.label_block_dict[false_label].add_parent(cur_block)

                cur_block = self.label_block_dict[next_label]
            cur_block.append(self.gen_code[index])

    def get_target_instr(self, instruction):
        op = instruction[0]
        op_without_type = op.split('_')[0]
        if (len(instruction) == 4) or \
           (len(instruction) == 3 and op_without_type != 'global') or \
           (len(instruction) == 2 and op_without_type not in ['alloc','fptosi','sitofp','define','param','read','print','return']):
            return instruction[-1]
        else:
            return None

    # check if instruction will generate kill or gen for Reaching definitions
    def instruction_has_rd_gen_kill(self, instruction):
        op = instruction[0]
        op_without_type = op.split('_')[0]
        if op_without_type in ['load','store','literal','elem','get','add','sub','mul','div','mod','and','or','not','ne','eq','lt','le','gt','ge'] or\
           (op == 'call' and len(instruction) == 3):
            return True
        else:
            return False

    # Gets defs for a temp
    def get_rd_defs(self, target):
        defs = []
        for label, block in self.label_block_dict.items():
            for instr_pos, instruction in enumerate(block.instructions):
                if self.get_target_instr(instruction) == target:
                    defs.append((label, instr_pos))

        return set(defs)

    def compute_rd_gen_kill(self, block):
        # gen[pn]  = gen[n]  U (gen[p] − kill[n])
        # kill[pn] = kill[p] U kill[n]

        target_defs_dict = {}

        for instr_pos, instruction in enumerate(block.instructions):
            if self.instruction_has_rd_gen_kill(instruction):
                target = self.get_target_instr(instruction)
                # Compute Kill
                if target not in target_defs_dict.keys():
                    target_defs_dict[target] = self.get_rd_defs(target)
                defs = target_defs_dict[target]
                kill          = defs - set([(block.label, instr_pos)])
                block.rd_kill = block.rd_kill.union(kill)
                # Compute Gen
                gen          = set([(block.label, instr_pos)])
                block.rd_gen = gen.union(block.rd_gen - kill)

    def compute_rd_in_out(self):
        # Initialize
        for label, block in self.label_block_dict.items():
            block.rd_out = set()

        # put all blocks into the changed set
        # B is all blocks in graph,
        changed_set = set(self.label_block_dict.values())

        # Iterate
        while (len(changed_set) != 0):
            # choose a block b in Changed;
            # remove it from the changed set
            block = changed_set.pop()

            # init IN[b] to be empty
            block.rd_in = set()

            # calculate IN[b] from predecessors' OUT[p]
            # for all blocks p in predecessors(b)
            for pred_block in block.parents:
                block.rd_in = block.rd_in.union(pred_block.rd_out)

            # save old OUT[b]
            old_out = block.rd_out

            # update OUT[b] using transfer function f_b(): OUT[b] = GEN[b] Union (IN[b] - KILL[b])
            block.rd_out = block.rd_gen.union(block.rd_in - block.rd_kill)

            # any change to OUT[b] compared to previous value?
            if (block.rd_out != old_out): # compare oldout vs. OUT[b]
                # if yes, put all successors of b into the changed set
                if isinstance(block, BasicBlock):
                    changed_set = changed_set.union(set([block.next_block]))
                elif isinstance(block, ConditionalBlock):
                    changed_set = changed_set.union(set([block.true_branch, block.false_branch]))

                # Check if None was inserted (block with next/branch to None)
                changed_set.discard(None)

    # return gen and kill from instruction (Liveness) -- return (gen_set, kill_set)
    def instruction_live_gen_kill(self, instruction):
        op = instruction[0]
        op_without_type = op.split('_')[0]
        # binary op
        if op_without_type in ['add','sub','mul','div','mod'] or op_without_type in ['ne','eq','lt','le','gt','ge','and','or','not']:
            return set([instruction[1], instruction[2]]), set([instruction[3]])
        # Params
        if op_without_type == 'param':
            return set([instruction[1]]), set()
        # Function call (if it has a target)
        if op == 'call':
            if len(instruction) == 3:
                return set(), set([instruction[2]])
            # TODO: add all global variables in gen set
            # called function might use all globals
        # Conditional Branch and Jump
        if op == 'cbranch':
            return set([instruction[1]]), set()
        # t <- b[i]
        if op_without_type == 'elem':
            return set([instruction[1], instruction[2]]), set([instruction[3]])
        if op_without_type in ['load','get','store','fptosi','sitofp']:
            return set([instruction[1]]), set([instruction[2]])
        # t <- C
        if op_without_type == 'literal':
            return set(), set([instruction[2]])
        # read(variable)
        if op_without_type == 'read':
            return set(), set([instruction[1]])
        # print(variable)
        if op_without_type == 'print':
            return set([instruction[1]]), set()
        # return x
        if op_without_type == 'return':
            op_type = op.split('_')[1]
            if op_type != 'void':
                return set([instruction[1]]), set()
        # Everything else
        return set(), set()


    def compute_live_gen_kill(self, block):
        '''
        Let's say P is the upper block and N is the lower block.
        Since the analysis is backwards, we have the previous result stored in
        N and the partial result of P.
        We will merge the two blocks (N and P) using the following equations:

        gen[PN] = (gen[N] - kill[P]) U gen[P]
        kill[PN] = (kill[N] - gen[P]) U kill[P]

        Why does this work?
        Suppose we have sets gen[N], kill[N] with the original gen and kill sets for N.
        Now suppose we have a new instruction, which corresponds to block P:

        i: int a = x;

        So we know that:
        kill[P] = {a}
        gen[P] = {x}

        Whatever is generated on N and P should be united in PN,
        except for everything that is killed in P.
        In the same way, whatever is killed on N but generated on
        P, shouldn't be in the new kill set anymore, unless it is
        killed in P as well.

        '''

        # For our case, we have:
        # block.live_gen  = (block.gen - kill) + gen
        # block.live_kill = (block.kill - gen) + kill

        # Compute Gen and Kill (backwards)
        for instr_pos, instruction in reversed(list(enumerate(block.instructions))):
            gen, kill       = self.instruction_live_gen_kill(instruction)
            block.live_gen  = (block.live_gen - kill).union(gen)
            block.live_kill = (block.live_kill - gen).union(kill)

    def compute_live_in_out(self):
        # Initialize
        for label, block in self.label_block_dict.items():
            block.live_in  = set()
            block.live_out = set()

        done = False
        # Iterate
        while (not done):
            done = True
            for label, block in self.label_block_dict.items():
                old_in  = block.live_in
                old_out = block.live_out

                block.live_in = block.live_gen.union(block.live_out - block.live_kill)

                if isinstance(block, BasicBlock):
                    if block.next_block:
                        block.live_out = block.next_block.live_in
                elif isinstance(block, ConditionalBlock):
                    if block.true_branch.live_in and block.false_branch.live_in:
                        block.live_out = block.true_branch.live_in.union(block.false_branch.live_in)
                    elif block.true_branch.live_in:
                        block.live_out = block.true_branch.live_in
                    else:
                        block.live_out = block.false_branch.live_in

                if (block.live_out != old_out or block.live_in != old_in):
                    done = False

    def clean_analysis(self):
        for label, block in self.label_block_dict.items():
            # Clean Reaching definition (rd) stuff
            block.rd_gen  = set()
            block.rd_kill = set()
            block.rd_in   = set()
            block.rd_out  = set()

            # Clean Liveness stuff
            block.live_gen  = set()
            block.live_kill = set()
            block.live_in   = set()
            block.live_out  = set()

    # Run Reaching Definitions and Liveness analysis
    def analyze(self):
        # Reaching Definitions
        for label, block in self.label_block_dict.items():
            self.compute_rd_gen_kill(block)
        self.compute_rd_in_out()

        # Liveness
        for label, block in self.label_block_dict.items():
            self.compute_live_gen_kill(block)
        self.compute_live_in_out()

    def dead_code_elimination(self):
        for label, block in self.label_block_dict.items():
            delete_indexes = set()
            # inner_live contains the set of variables that are alive through the block
            # (initially contains only variables used after this block)
            inner_live = block.live_out
            for instr_pos, instruction in reversed(list(enumerate(block.instructions))):
                uses, defs = self.instruction_live_gen_kill(instruction)
                dead = False
                # defs has always only one element so this is not a real loop
                for d in defs:
                    '''
                    If we are defining something that is not used, then mark this code as dead and
                    don't even save their uses, because we might find more dead code from this deletion.
                    '''
                    if not d in inner_live:
                        dead = True
                        delete_indexes.add(instr_pos)
                # if the instruction is not dead, we kill the definitions and add uses to the inner_live set
                if not dead:
                    inner_live = inner_live - defs
                    inner_live = inner_live.union(uses)

            # remove dead code from delete_indexes list
            updated_instructions = []
            for i,inst in enumerate(block.instructions):
                if not i in delete_indexes:
                    updated_instructions.append(inst)
            block.instructions = updated_instructions
        # TODO dead code elimination for unused allocs

    def instruction_is_binary_op(self, instruction):
        op = instruction[0]
        op_without_type = op.split('_')[0]
        #TODO: check if 'elem','get' should be included
        if op_without_type in ['add','sub','mul','div','mod','and','or','not','ne','eq','lt','le','gt','ge']:
            return True
        else:
            return False

    def fold_instruction(self, instruction, left_op, right_op):
        op = instruction[0]
        op_without_type = op.split('_')[0]
        instr_type = op.split('_')[1]
        target = instruction[3]

        #TODO: char and pointers
        if instr_type == 'int':
            left_op, right_op = int(left_op), int(right_op)
        elif instr_type == 'float':
            left_op, right_op = float(left_op), float(right_op)

        fold = op_lambdas[op_without_type](left_op, right_op)

        return ('literal_{}'.format(instr_type), fold, target)

    def constant_propagation_and_folding(self):
        for label, block in self.label_block_dict.items():
            temp_constant_dict = {}

            for pred_block in block.parents:
                for block_label, instr_index in pred_block.rd_out:
                    instruction = self.label_block_dict[block_label].instructions[instr_index]
                    op = instruction[0]
                    target = self.get_target_instr(instruction)
                    op_without_type = op.split('_')[0]
                    if op_without_type == 'literal':
                        literal = instruction[1]
                        temp_constant_dict[target] = literal

            for instr_pos, instruction in enumerate(block.instructions):
                op = instruction[0]
                target = self.get_target_instr(instruction)
                op_without_type = op.split('_')[0]

                if op_without_type in ['store','load']:
                    source = instruction[1]
                    if source in temp_constant_dict:
                        instr_type = op.split('_')[1]
                        new_op = 'literal_{}'.format(instr_type)
                        new_instruction = (new_op, temp_constant_dict[source], instruction[2])
                        block.instructions[instr_pos] = new_instruction
                        # print(instruction, '->', new_instruction)
                        instruction = block.instructions[instr_pos]
                        op = instruction[0]
                        target = self.get_target_instr(instruction)
                        op_without_type = op.split('_')[0]

                if self.instruction_is_binary_op(instruction):
                    left_op  = instruction[1]
                    right_op = instruction[2]
                    if left_op in temp_constant_dict.keys() and right_op in temp_constant_dict.keys():
                        left_op  = temp_constant_dict[instruction[1]]
                        right_op = temp_constant_dict[instruction[2]]
                        new_instruction = self.fold_instruction(instruction, left_op, right_op)
                        print(instruction, '->', new_instruction)
                        block.instructions[instr_pos] = new_instruction

                        instruction = block.instructions[instr_pos]
                        op = instruction[0]
                        target = self.get_target_instr(instruction)
                        op_without_type = op.split('_')[0]

                if op_without_type == 'literal':
                    # save constant
                    literal = instruction[1]
                    temp_constant_dict[target] = literal
                elif self.instruction_has_rd_gen_kill(instruction):
                    if target in temp_constant_dict:
                        temp_constant_dict.pop(target)

            # TODO: Improve, only recompute analysis if necessary
            self.compute_rd_in_out()

    def optimize(self):
        self.constant_propagation_and_folding()
        self.dead_code_elimination()


class CFG_Program():
    def __init__(self, gen_code):
        self.gen_code      = gen_code
        self.func_cfg_dict = {}

        self.create_cfgs()

    def create_cfgs(self):
        function_code_dict = {}

        self.gen_code.append(('define',''))

        cur_function   = '__global'
        start_function = 0
        for instr_pos, instruction in enumerate(self.gen_code):
            if instruction[0] == 'define':
                function_code = self.gen_code[start_function:instr_pos]
                self.func_cfg_dict[cur_function] = CFG(function_code)
                cur_function   = instruction[1]
                start_function = instr_pos

        del self.gen_code[-1]

    def output(self, ir_filename=None):
        aux = sys.stdout
        if ir_filename:
            print("Outputting CFG to %s" % ir_filename)
            sys.stdout = open(ir_filename, 'w')

        for function, cfg in self.func_cfg_dict.items():
            print('====================== ' + function + ' ======================')
            cfg.output()

        for function, cfg in self.func_cfg_dict.items():
            print('====================== RD ANALYSIS for ' + function + ' ======================')
            cfg.output_rd()
            print('\n\n\n')

        for function, cfg in self.func_cfg_dict.items():
            print('====================== LIVENESS ANALYSIS for ' + function + ' ======================')
            cfg.output_liveness()
            print('\n\n\n')
        sys.stdout = aux

    def output_optimized_code(self, ir_filename=None):
        aux = sys.stdout
        if ir_filename:
            print("Outputting CFG to %s" % ir_filename)
            sys.stdout = open(ir_filename, 'w')
        for _, cfg in self.func_cfg_dict.items():
            cfg.output_optimized_code()
        sys.stdout = aux

    def clean_analysis(self):
        for _, cfg in self.func_cfg_dict.items():
            cfg.clean_analysis()

    def optimize(self):
        self.clean_analysis()
        for _, cfg in self.func_cfg_dict.items():
            # TODO analyse and optimize several times
            # TODO do we need to analyse each time we optimize again? I think we do.
            cfg.analyze()
            cfg.optimize()

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
    # perform optimizations
    cfg.optimize()
    cfg.output()
    # output result of CFG to file
    cfg_filename = filename[:-3] + '.cfg'
    cfg.output(cfg_filename)
    opt_filename = filename[:-3] + '.opt'
    cfg.output_optimized_code(opt_filename)

    interpreter = Interpreter()
    interpreter.run(gencode.code)
