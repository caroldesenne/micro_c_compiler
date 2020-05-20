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
        self.gen_code         = gen_code
        self.label_block_dict = {}
        self.label_blocktype_dict = {}
        self.first_block      = None
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

    def output(self, ir_filename=None):
        '''
        outputs generated IR code to given file. If no file is given, output to stdout
        '''
        if ir_filename:
            print("Outputting CFG to %s." % ir_filename)
            buf = open(ir_filename, 'w')
        else:
            print("Printing CFG:\n\n")
            buf = sys.stdout

        for k, v in self.label_block_dict.items():
            pprint('-------------------- Block {}:  --------------------'.format(k),buf)
            for code in v.instructions:
                    pprint(code,buf)
            try:
                pprint('------ Parents -------',buf)
                for parent in v.parents:
                    pprint(parent.label,buf)
                if isinstance(v, ConditionalBlock):
                    pprint('---- true branch label', v.true_branch.label, 'false branch label', v.false_branch.label, '----',buf)
                elif isinstance(v, BasicBlock):
                    pprint('--------------------- jump', v.next_block.label, '----------------------',buf)
            except:
                pass

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
            else:
                cur_block.append(self.gen_code[index])

    # check if instruction will generate kill or gen
    # TODO: Improve, check all cases
    def instruction_has_gen_kill(self, instruction):
        op = instruction[0]
        instr_without_type = op.split('_')[0]
        if instr_without_type in ['load','store','literal','elem','get','add','sub','mul','div','mod'] or op == 'call':
            return True
        else:
            return False

    # Gets defs for a temp
    def get_defs(self, target):
        defs = []
        for label, block in self.label_block_dict.items():
            for instr_pos, instruction in enumerate(block.instructions):
                if self.get_target_instr(instruction) == target:
                    defs.append((label, instr_pos))

        return set(defs)

    def get_target_instr(self, instruction):
        if (len(instruction) == 4) or \
           (len(instruction) == 3 and instruction != 'global_type') or \
           (len(instruction) == 2 and instruction not in ['alloc_type','fptosi','sitofp','define','param_type','read_type','print_type']):
            return instruction[-1]
        else:
            return None

    def compute_rd_gen_kill(self, block):
        # gen[pn]  = gen[n]  U (gen[p] − kill[n])
        # kill[pn] = kill[p] U kill[n]

        # Compute Kill
        for instr_pos, instruction in enumerate(block.instructions):
            if self.instruction_has_gen_kill(instruction):
                target = self.get_target_instr(instruction)
                if target is not None:
                    defs   = self.get_defs(target)
                    kill   = defs - set([(block.label, instr_pos)])
                    block.rd_kill = block.rd_kill.union(kill)

        # Compute Gen
        for instr_pos, instruction in enumerate(block.instructions):
            if self.instruction_has_gen_kill(instruction):
                target = self.get_target_instr(instruction)
                gen    = set([(block.label, instr_pos)])
                # TODO: Check, not sure if we can simply use kill from entire block
                # Expand formula for 3 lines, just to be sure
                block.rd_gen = block.rd_gen.union(gen - block.rd_kill)

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



    def optimize(self):
        for label, block in self.label_block_dict.items():
            self.compute_rd_gen_kill(block)
        self.compute_rd_in_out()


        print('=========================== GEN and KILL ===========================')
        for label, block in self.label_block_dict.items():
            print('Block ', label)
            print('GEN : ', block.rd_gen)
            print('KILL: ', block.rd_kill)

        print('============================ IN and OUT ============================')
        for label, block in self.label_block_dict.items():
            print('Block ', label)
            print('IN : ', block.rd_in)
            print('OUT: ', block.rd_out)


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

    cfg = CFG(gencode.code)
    # remove this print in the future
    cfg.print()
    # output result of CFG to file
    cfg_filename = filename[:-3] + '.cfg'
    cfg.output(cfg_filename)

    # perform optimizations
    cfg.optimize()