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
        # Not all instructions that have length 1 are labels. Ohter option is: ('return_void',)
        if (len(instruction) == 1) and (instruction[0]!='return_void'):
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
        aux = sys.stdout
        if ir_filename:
            print("Outputting CFG to %s." % ir_filename)
            sys.stdout = open(ir_filename, 'w')
    
        for k, v in self.label_block_dict.items():
            print()
            print("Block {}".format(k))
            for code in v.instructions:
                print("    "+str(code))
            if isinstance(v, BasicBlock) and v.next_block:
                print("Next block {}".format(v.next_block.label))
        sys.stdout = aux

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
    def gen_kill_not_empty(self, instruction):
        if len(instruction[0]) < 3:
            return False
        else:
            return True

    # Gets defs for a temp
    def get_defs(self, target):
        defs = []
        for label, block in self.label_block_dict.items():
            for instr_pos, instruction in enumerate(block.instructions):
                if self.get_target_instr(instruction) == target:
                    defs.append((label, instr_pos))

        return set(defs)

    # TODO: Improve, check all cases
    def get_target_instr(self, instruction):
        if len(instruction) == 3 or len(instruction) == 4:
            return instruction[-1]
        else:
            return None

    def compute_rd_gen_kill(self, block):
        # gen[pn]  = gen[n]  U (gen[p] − kill[n])
        # kill[pn] = kill[p] U kill[n]

        # Compute Kill
        for instr_pos, instruction in enumerate(block.instructions):
            if self.gen_kill_not_empty(instruction):
                target = self.get_target_instr(instruction)
                if target is not None:
                    defs   = self.get_defs(target)
                    kill   = defs - set([(block.label, instr_pos)])
                    block.rd_kill = block.rd_kill.union(kill)

        # Compute Gen
        for instr_pos, instruction in enumerate(block.instructions):
            if self.gen_kill_not_empty(instruction):
                target = self.get_target_instr(instruction)
                gen    = set([(block.label, instr_pos)])
                # TODO: Check, not sure if we can simply use kill from entire block
                # Expand formula for 3 lines, just to be sure
                block.rd_gen = block.rd_gen.union(gen - block.rd_kill)


    def optimize(self):
        for label, block in self.label_block_dict.items():
            self.compute_rd_gen_kill(block)
            print(label, block.rd_gen, block.rd_kill)

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
    #cfg.print()
    cfg.output()
    # output result of CFG to file
    cfg_filename = filename[:-3] + '.cfg'
    cfg.output(cfg_filename)
    
    # perform optimizations
    cfg.optimize()
