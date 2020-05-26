import sys
from pprint import pprint
from parser import Parser
from ast import *
from checker import CheckProgramVisitor
from generator import GenerateCode

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

        # Reaching definition (rd) stuff
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
        if op_without_type in ['load','store','literal','elem','get','add','sub','mul','div','mod'] or\
           op_without_type in ['ne','eq','lt','le','gt','ge'] or\
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
        if op_without_type in ['add','sub','mul','div','mod'] or op_without_type in ['ne','eq','lt','le','gt','ge']:
            return set([instruction[1], instruction[2]]), set([instruction[3]])
        # Params
        if op_without_type == 'param':
            return set([instruction[1]]), set()
        # Function call (if it has a target)
        if op == 'call' and len(instruction) == 3:
            return set(), set([instruction[2]])
        # Conditional Branch and Jump
        if op == 'cbranch' or op == 'jump':
            return set([instruction[1]]), set()
        # t <- M[b[i]]
        if op_without_type == 'elem':
            return set([instruction[1], instruction[2]]), set([instruction[3]])
        if op_without_type in ['load','get']:
            return set([instruction[1]]), set([instruction[2]])
        # M[a] <- b
        if op_without_type == 'store':
            return set([instruction[1]]), set([instruction[2]])
        # t <- C
        if op_without_type == 'literal':
            return set(), set([instruction[2]])
        # return x
        if op_without_type == 'return':
            op_type = op.split('_')[1]
            if op_type != 'void':
                return set([instruction[1]]), set()
        # Everything else
        return set(), set()


    def compute_live_gen_kill(self, block):
        # gen[pn]  = gen[p]  U (gen[n] − kill[p])
        # kill[pn] = kill[p] U kill[n]

        # Compute Gen and Kill (backwards)
        for instr_pos, instruction in reversed(list(enumerate(block.instructions))):
            gen, kill       = self.instruction_live_gen_kill(instruction)
            block.live_gen  = block.live_gen.union(gen - block.live_kill)
            block.live_kill = block.live_kill.union(kill)

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

    def optimize(self):
        pass

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
            print("Outputting CFG to %s." % ir_filename)
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

    def optimize(self):
        for _, cfg in self.func_cfg_dict.items():
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
    # remove this print in the future
    #cfg.print()
    cfg.optimize()
    cfg.output()
    # output result of CFG to file
    cfg_filename = filename[:-3] + '.cfg'
    cfg.output(cfg_filename)

    # perform optimizations
    # cfg.optimize()