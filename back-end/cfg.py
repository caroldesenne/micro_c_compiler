class Block(object):
    def __init__(self, label):
        self.instructions = []   # Instructions in the block
        self.next_block   = None # Link to the next block
        self.label        = label

    def append(self,instr):
        self.instructions.append(instr)

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
        self.first_block      = None

        self.create_blocks()

    '''
    Get type from instruction (basically if label or not) and return label converted to int.
    '''
    def get_code_type(self, code):
        op = code[0]
        if len(code) == 1:
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

    def print(self):
        for k, v in self.label_block_dict.items():
            try:
                print('-------------------- Block {}:  --------------------'.format(k))
                if isinstance(v, ConditionalBlock):
                    for code in v.instructions:
                        print(code)
                    print('---- true branch label', v.true_branch.label, 'false branch label', v.false_branch.label, '----')
                else:
                    for code in v.instructions:
                        print(code)
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
        label_blocktype_dict = {}

        cur_label = 0
        for index in range(len(self.gen_code)):
            op = self.get_code_type(self.gen_code[index])
            if isinstance(op, int):
                prev_op = self.get_code_type(self.gen_code[index-1])
                if prev_op == 'jump' or prev_op == 'cbranch':
                    label_blocktype_dict[cur_label] = prev_op
                else:
                    label_blocktype_dict[cur_label] = 'basic'
                cur_label = op

        # remove  dummy label at the end
        del self.gen_code[-1]


        ##########################################################
        # Second iteration, creating the blocks and linking them #
        ##########################################################

        # Create first block
        self.first_block = self.new_block(label_blocktype_dict[0], 0)
        self.label_block_dict[0] = self.first_block

        cur_block = self.first_block
        for index in range(len(self.gen_code)):
            op = self.get_code_type(self.gen_code[index])
            # if we find a label, it means we are done with previous block
            if isinstance(op, int):
                # creates next block
                next_label = op
                if next_label not in self.label_block_dict.keys():
                    self.label_block_dict[next_label] = self.new_block(label_blocktype_dict[next_label], next_label)

                # link current block (next/jump/branch)
                if label_blocktype_dict[cur_block.label] == 'basic':
                    cur_block.next_block = self.label_block_dict[next_label]
                elif label_blocktype_dict[cur_block.label] == 'jump':
                    # dirty way to get jump label from instruction
                    jump_label = int(self.gen_code[index-1][1][1:])
                    if jump_label not in self.label_block_dict.keys():
                        self.label_block_dict[jump_label] = self.new_block(label_blocktype_dict[jump_label], jump_label)
                    cur_block.next_block = self.label_block_dict[jump_label]
                elif label_blocktype_dict[cur_block.label] == 'cbranch':
                    true_label  = int(self.gen_code[index-1][2][1:])
                    if true_label not in self.label_block_dict.keys():
                        self.label_block_dict[true_label] = self.new_block(label_blocktype_dict[true_label], true_label)
                    cur_block.true_branch  = self.label_block_dict[true_label]

                    false_label = int(self.gen_code[index-1][3][1:])
                    if false_label not in self.label_block_dict.keys():
                        self.label_block_dict[false_label] = self.new_block(label_blocktype_dict[false_label], false_label)
                    cur_block.false_branch = self.label_block_dict[false_label]

                cur_block = self.label_block_dict[next_label]
            else:
                cur_block.append(self.gen_code[index])

if __name__ == "__main__":
    # hardcoded bubble sort example for now
    gc = [
        ('global_string', '@.str.0', 'Enter number of elements: '),
        ('global_string', '@.str.1', 'Enter '),
        ('global_string', '@.str.2', ' integers\\n'),
        ('global_string', '@.str.3', 'Sorted list in ascending order:\\n'),
        ('global_string', '@.str.4', ' '),
        ('define', '@main'),
        ('alloc_int_100', '%2'),
        ('alloc_int', '%3'),
        ('alloc_int', '%4'),
        ('alloc_int', '%5'),
        ('alloc_int', '%6'),
        ('print_string', '@.str.0'),
        ('read_int', '%7'),
        ('store_int', '%7', '%3'),
        ('print_string', '@.str.1'),
        ('load_int', '%3', '%8'),
        ('print_int', '%8'),
        ('print_string', '@.str.2'),
        ('literal_int', 0, '%12'),
        ('store_int', '%12', '%4'),
        ('9',),
        ('load_int', '%4', '%13'),
        ('load_int', '%3', '%14'),
        ('lt_int', '%13', '%14', '%15'),
        ('cbranch', '%15', '%10', '%11'),
        ('10',),
        ('load_int', '%4', '%16'),
        ('elem_int', '%2', '%16', '%17'),
        ('read_int', '%18'),
        ('store_int_*', '%18', '%17'),
        ('load_int', '%4', '%19'),
        ('literal_int', 1, '%20'),
        ('add_int', '%19', '%20', '%21'),
        ('store_int', '%21', '%4'),
        ('jump', '%9'),
        ('11',),
        ('literal_int', 0, '%25'),
        ('store_int', '%25', '%4'),
        ('22',),
        ('literal_int', 1, '%26'),
        ('load_int', '%3', '%27'),
        ('sub_int', '%27', '%26', '%28'),
        ('load_int', '%4', '%29'),
        ('lt_int', '%29', '%28', '%30'),
        ('cbranch', '%30', '%23', '%24'),
        ('23',),
        ('literal_int', 0, '%34'),
        ('store_int', '%34', '%5'),
        ('31',),
        ('load_int', '%3', '%35'),
        ('load_int', '%4', '%36'),
        ('sub_int', '%35', '%36', '%37'),
        ('literal_int', 1, '%38'),
        ('sub_int', '%37', '%38', '%39'),
        ('load_int', '%5', '%40'),
        ('lt_int', '%40', '%39', '%41'),
        ('cbranch', '%41', '%32', '%33'),
        ('32',),
        ('load_int', '%5', '%45'),
        ('elem_int', '%2', '%45', '%46'),
        ('literal_int', 1, '%47'),
        ('load_int', '%5', '%48'),
        ('add_int', '%48', '%47', '%49'),
        ('elem_int', '%2', '%49', '%50'),
        ('load_int_*', '%46', '%51'),
        ('load_int_*', '%50', '%52'),
        ('gt_int', '%51', '%52', '%53'),
        ('cbranch', '%53', '%42', '%43'),
        ('42',),
        ('load_int', '%5', '%54'),
        ('elem_int', '%2', '%54', '%55'),
        ('load_int_*', '%55', '%56'),
        ('store_int', '%56', '%6'),
        ('literal_int', 1, '%57'),
        ('load_int', '%5', '%58'),
        ('add_int', '%58', '%57', '%59'),
        ('elem_int', '%2', '%59', '%60'),
        ('load_int_*', '%60', '%61'),
        ('load_int', '%5', '%62'),
        ('elem_int', '%2', '%62', '%63'),
        ('store_int_*', '%61', '%63'),
        ('load_int', '%6', '%64'),
        ('literal_int', 1, '%65'),
        ('load_int', '%5', '%66'),
        ('add_int', '%66', '%65', '%67'),
        ('elem_int', '%2', '%67', '%68'),
        ('store_int_*', '%64', '%68'),
        ('43',),
        ('load_int', '%5', '%69'),
        ('literal_int', 1, '%70'),
        ('add_int', '%69', '%70', '%71'),
        ('store_int', '%71', '%5'),
        ('jump', '%31'),
        ('33',),
        ('load_int', '%4', '%72'),
        ('literal_int', 1, '%73'),
        ('add_int', '%72', '%73', '%74'),
        ('store_int', '%74', '%4'),
        ('jump', '%22'),
        ('24',),
        ('print_string', '@.str.3'),
        ('literal_int', 0, '%78'),
        ('store_int', '%78', '%4'),
        ('75',),
        ('load_int', '%4', '%79'),
        ('load_int', '%3', '%80'),
        ('lt_int', '%79', '%80', '%81'),
        ('cbranch', '%81', '%76', '%77'),
        ('76',),
        ('load_int', '%4', '%82'),
        ('elem_int', '%2', '%82', '%83'),
        ('load_int_*', '%83', '%84'),
        ('print_int', '%84'),
        ('print_string', '@.str.4'),
        ('load_int', '%4', '%85'),
        ('literal_int', 1, '%86'),
        ('add_int', '%85', '%86', '%87'),
        ('store_int', '%87', '%4'),
        ('jump', '%75'),
        ('77',),
        ('literal_int', 0, '%88'),
        ('store_int', '%88', '%0'),
        ('jump', '%1'),
        ('1',),
        ('load_int', '%0', '%89'),
        ('return_int', '%89'),
    ]
    cfg = CFG(gc)

    cfg.print()
