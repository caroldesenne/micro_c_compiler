import ply.yacc as yacc
from lexer import Lexer
from uc_ast import *

class Parser():

    def _type_modify_decl(self, decl, modifier):
        """ Tacks a type modifier on a declarator, and returns
            the modified declarator.
            Note: the declarator and modifier may be modified
        """
        modifier_head = modifier
        modifier_tail = modifier

        # The modifier may be a nested list. Reach its tail.
        while modifier_tail.type:
            modifier_tail = modifier_tail.type

        # If the decl is a basic type, just tack the modifier onto it
        if isinstance(decl, VarDecl):
            modifier_tail.type = decl
            return modifier
        else:
            # Otherwise, the decl is a list of modifiers. Reach
            # its tail and splice the modifier onto the tail,
            # pointing to the underlying basic type.
            decl_tail = decl

            while not isinstance(decl_tail.type, VarDecl):
                decl_tail = decl_tail.type

            modifier_tail.type = decl_tail.type
            decl_tail.type = modifier_head
            return decl

    def _fix_decl_name_type(self, decl, typename):
        """ Fixes a declaration. Modifies decl.
        """
        # Reach the underlying basic type
        t = decl
        while not isinstance(t, VarDecl):
            t = t.type

        decl.name = t.name

        # The typename is a list of types. If any type in this
        # list isn't an Type, it must be the only
        # type in the list.
        # If all the types are basic, they're collected in the
        # Type holder.
        t.type = typename
        return decl

    def _build_declarations(self, spec, decls):
        """ Builds a list of declarations all sharing the given specifiers.
        """
        declarations = []

        for decl in decls:
            assert decl['decl'] is not None
            declaration = Decl(
                    name=None,
                    type=decl['decl'],
                    init=decl.get('init'),
                    coord=decl['decl'].coord)

            fixed_decl = self._fix_decl_name_type(declaration, spec)
            declarations.append(fixed_decl)

        return declarations

    def _token_coord(self, p, token_idx):
        last_cr = p.lexer.lexer.lexdata.rfind('\n', 0, p.lexpos(token_idx))
        if last_cr < 0:
            last_cr = -1
        column = (p.lexpos(token_idx) - (last_cr))
        return Coord(p.lineno(token_idx), column)

    def print_error(msg, x, y):
        print('Lexical error: %s at %d:%d' %(msg,x,y))

    def __init__(self):
        self.lexer = Lexer(error_func=self.print_error)
        self.lexer.build()
        self.tokens = self.lexer.tokens
        self.parser = yacc.yacc(module=self, start='program')

    def parse(self, data, filename='', deb=False):
        return self.parser.parse(input=data, lexer=self.lexer, debug=deb)

    def p_program(self, p):
        """ program  : global_declaration_list
        """
        p[0] = Program(p[1],coord=self._token_coord(p, 1))

    def p_global_declaration_list(self, p):
        """
        global_declaration_list : global_declaration
                                | global_declaration_list global_declaration
        """
        if len(p)==2:
            p[0] = [p[1]]
        else:
            p[0] = p[1]+[p[2]]

    def p_global_declaration1(self, p):
        """
        global_declaration : declaration
        """
        p[0] = GlobalDecl(p[1],coord=self._token_coord(p, 1))

    def p_global_declaration2(self, p):
        """
        global_declaration : function_definition
        """
        p[0] = p[1]

    def p_declaration_list(self, p):
        """
        declaration_list : declaration
                         | declaration_list declaration
        """
        if len(p)==2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_declaration_list_opt(self, p):
        """
        declaration_list_opt : declaration_list
                             | empty
        """
        p[0] = p[1]

    def p_function_definition(self, p):
        """
        function_definition : type_specifier declarator declaration_list_opt compound_statement
        """
        decl = p[2]
        decls = self._build_declarations(spec=p[1],decls=[dict(decl=decl, init=None)])[0]
        p[0] = FuncDef(decl=decls, type=p[1], dl=p[3], cs=p[4],coord=p[2].coord)

    def p_function_definition1(self, p):
        """
        function_definition : declarator declaration_list_opt compound_statement
        """
        t = Type(['int'],coord=self._token_coord(p, 1))
        decl = p[1]
        decls = self._build_declarations(spec=t,decls=[dict(decl=decl, init=None)])[0]
        p[0] = FuncDef(decls, None, p[2], p[3],coord=p[1].coord)

    def p_identifier(self, p):
        """
        identifier : ID
        """
        p[0] = ID(p[1],coord=self._token_coord(p, 1))

    def p_string(self, p):
        """
        string : STRING_LITERAL
        """
        p[0] = Constant('string',p[1],coord=self._token_coord(p, 1))

    def p_integer_constant(self, p):
        """
        integer_constant : INT_CONST
        """
        p[0]= Constant('int',p[1],coord=self._token_coord(p, 1))

    def p_character_constant(self, p):
        """
        character_constant : CHAR_CONST
        """
        p[0]= Constant('char',p[1],coord=self._token_coord(p, 1))

    def p_floating_constant(self, p):
        """
        floating_constant : FLOAT_CONST
        """
        p[0]= Constant('float',p[1],coord=self._token_coord(p, 1))

    def p_type_specifier(self, p):
        """
        type_specifier : VOID
                       | CHAR
                       | INT
                       | FLOAT
        """
        p[0] = Type([p[1]],coord=self._token_coord(p, 1))

    def p_identifier_list(self, p):
        """
        identifier_list : identifier
                        | identifier_list identifier
        """
        if len(p)==2:
            p[0] = [p[1]]
        else:
            p[0] = p[1]+[p[2]]

    def p_identifier_list_opt(self, p):
        """
        identifier_list_opt : identifier_list
                            | empty
        """
        p[0] = p[1]

    def p_declarator(self, p):
        """
        declarator : direct_declarator
                   | pointer direct_declarator
        """
        if len(p)==2:
            p[0] = p[1]
        else:
            p[0] = self._type_modify_decl(p[2],p[1])

    def p_pointer(self, p):
        """
        pointer : TIMES
                | TIMES pointer
        """
        if len(p)==2:
            p[0] = PtrDecl([p[1]],self._token_coord(p, 1))
        else:
            p[2].stars.append(p[1])
            p[0] = p[2]

    def p_direct_declarator(self, p):
        """
        direct_declarator : identifier
                          | LPAREN declarator RPAREN
                          | direct_declarator LBRACKET constant_expression RBRACKET
        """
        if len(p)==2:
            p[0] = VarDecl(p[1],coord=p[1].coord)
        elif len(p)==4:
            p[0] = p[2]
        elif len(p)==5:
            arr = ArrayDecl(None,p[3],coord=p[1].coord)
            p[0] = self._type_modify_decl(decl=p[1],modifier=arr)


    def p_direct_declarator2(self, p):
        """
        direct_declarator : direct_declarator LBRACKET RBRACKET
                          | direct_declarator LPAREN parameter_list RPAREN
        """
        if len(p)==4:
            arr = ArrayDecl(None,None,coord=p[1].coord)
            p[0] = self._type_modify_decl(decl=p[1],modifier=arr)
        elif len(p)==5:
            func = FuncDecl(args=p[3],type=None,coord=p[1].coord)
            p[0] = self._type_modify_decl(decl=p[1], modifier=func)

    def p_direct_declarator3(self, p):
        """
        direct_declarator : direct_declarator LPAREN identifier_list_opt RPAREN
        """
        func = FuncDecl(args=p[3],type=None,coord=p[1].coord)
        p[0] = self._type_modify_decl(decl=p[1], modifier=func)

    def p_constant_expression(self, p):
        """
        constant_expression : binary_expression
        """
        p[0] = p[1]

    def p_binary_expression(self, p):
        """
        binary_expression : cast_expression
                          | binary_expression TIMES binary_expression
                          | binary_expression DIVIDE binary_expression
                          | binary_expression MOD binary_expression
                          | binary_expression PLUS binary_expression
                          | binary_expression MINUS binary_expression
                          | binary_expression LT binary_expression
                          | binary_expression LE binary_expression
                          | binary_expression GT binary_expression
                          | binary_expression GE binary_expression
                          | binary_expression EQ binary_expression
                          | binary_expression NE binary_expression
                          | binary_expression AND binary_expression
                          | binary_expression OR binary_expression
        """
        if len(p)==2:
            p[0] = p[1]
        else:
            p[0] = BinaryOp(p[2], p[1], p[3],coord=p[1].coord)

    def p_cast_expression(self, p):
        """
        cast_expression : unary_expression
                        | LPAREN type_specifier RPAREN cast_expression
        """
        if len(p)==2:
            p[0] = p[1]
        else:
            p[0] = Cast(p[2],p[4])

    def p_unary_expression(self, p):
        """
        unary_expression : postfix_expression
                         | PLUSPLUS unary_expression
                         | MINUSMINUS unary_expression
                         | unary_operator cast_expression
        """
        if len(p)==2:
            p[0] = p[1]
        else:
            p[0] = UnaryOp(p[1],p[2],coord=p[2].coord)

    def p_postfix_expression(self, p):
        """
        postfix_expression : primary_expression
                           | postfix_expression PLUSPLUS
                           | postfix_expression MINUSMINUS
                           | postfix_expression LPAREN expression_opt RPAREN
        """
        if len(p)==2:
            p[0] = p[1]
        elif len(p)==3:
            p[0] = UnaryOp('p'+p[2],p[1],coord=p[1].coord)
        else:
            p[0] = FuncCall(p[1],p[3],coord=p[1].coord)

    def p_postfix_expression2(self, p):
        """
        postfix_expression : postfix_expression LBRACKET expression RBRACKET
        """
        p[0] = ArrayRef(p[1],p[3],coord=p[1].coord)


    def p_primary_expression(self, p):
        """
        primary_expression : identifier
                           | constant
                           | string
                           | LPAREN expression RPAREN
        """
        if len(p)==2: # (id,constant and string)
            p[0] = p[1]
        else: # (expression)
            p[0] = p[2]

    def p_constant(self, p):
        """
        constant : integer_constant
                 | character_constant
                 | floating_constant
        """
        p[0] = p[1]

    def p_expression(self, p):
        """
        expression : assignment_expression
                   | expression COMMA assignment_expression
        """
        if len(p) == 2: # single expression
            p[0] = ExprList([p[1]], coord=p[1].coord)
        else:
            p[1].list.append(p[3])
            p[0] = p[1]

    def p_expression_opt(self, p):
        """
        expression_opt : expression
                       | empty
        """
        p[0] = p[1]

    def p_expression_statement(self, p):
        """
        expression_statement : expression_opt SEMI
        """
        if p[1] is None:
            p[0] = EmptyStatement(coord=self._token_coord(p, 1))
        else:
            p[0] = p[1]

    def p_assignment_expression(self, p):
        """
        assignment_expression : binary_expression
                              | unary_expression assignment_operator assignment_expression
        """
        if len(p)==2:
            p[0] = p[1]
        else:
            p[0] = Assignment(p[2],p[1],p[3],p[1].coord)

    def p_assignment_operator(self, p):
        """
        assignment_operator : EQUALS
                            | TIMESEQUAL
                            | DIVEQUAL
                            | MODEQUAL
                            | PLUSEQUAL
                            | MINUSEQUAL
        """
        p[0] = p[1]

    def p_unary_operator(self, p):
        """
        unary_operator : ADDRESS
                       | TIMES
                       | PLUS
                       | MINUS
                       | NOT
        """
        p[0] = p[1]

    def p_parameter_list(self, p):
        """
        parameter_list : parameter_declaration
                       | parameter_list COMMA parameter_declaration
        """
        if len(p) == 2: # single parameter
            p[0] = ParamList([p[1]], p[1].coord)
        else:
            p[1].list.append(p[3])
            p[0] = p[1]

    def p_parameter_declaration(self, p):
        """
        parameter_declaration : type_specifier declarator
        """
        p[0] = self._build_declarations(spec=p[1],decls=[dict(decl=p[2])])[0]

    def p_init_declarator_list(self, p):
        """
        init_declarator_list : init_declarator
                             | init_declarator_list COMMA init_declarator
        """
        if len(p)==2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_init_declarator_list_opt(self, p):
        """
        init_declarator_list_opt : init_declarator_list
                                 | empty
        """
        p[0] = p[1]

    def p_decl_body(self, p):
        """
        decl_body : type_specifier init_declarator_list_opt
        """
        if p[2] is None:
            decls = [Decl(name=None,type=p[1],init=None,coord=p[1].coord)]
        else:
            decls = self._build_declarations(spec=p[1],decls=p[2])
        p[0] = decls

    def p_declaration(self, p):
        """
        declaration : decl_body SEMI
        """
        p[0] = p[1]

    def p_init_declarator(self, p):
        """
        init_declarator : declarator
                        | declarator EQUALS initializer
        """
        p[0] = dict(decl=p[1], init=(p[3] if len(p) > 2 else None))

    def p_initializer(self, p):
        """
        initializer : assignment_expression
                    | LBRACE initializer_list RBRACE
                    | LBRACE initializer_list COMMA RBRACE
        """
        if len(p)==2:
            p[0] = p[1]
        else:
            p[0] = InitList(p[2],coord=p[2][0].coord)

    def p_initializer_list(self, p):
        """
        initializer_list : initializer
                         | initializer_list COMMA initializer
        """
        if len(p)==2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_block_item(self, p):
        """ block_item  : declaration
                        | statement
        """
        p[0] = p[1] if isinstance(p[1], list) else [p[1]]

    def p_block_item_list(self, p):
        """ block_item_list : block_item
                            | block_item_list block_item
        """
        # Empty block items (plain ';') produce [None], so ignore them
        p[0] = p[1] if (len(p) == 2 or p[2] == [None]) else p[1] + p[2]

    def p_block_item_list_opt(self, p):
        """ block_item_list_opt : block_item_list
                                | empty
        """
        p[0] = p[1]

    def p_compound_statement(self, p):
        """
        compound_statement : LBRACE block_item_list_opt RBRACE
        """
        p[0] = Compound(p[2],coord=self._token_coord(p, 1))

    def p_statement(self, p):
        """
        statement : expression_statement
               	  | compound_statement
               	  | selection_statement
               	  | jump_statement
                  | assert_statement
                  | print_statement
                  | read_statement
                  | iteration_statement
        """
        p[0] = p[1]

    def p_selection_statement(self, p):
        """
        selection_statement : IF LPAREN expression RPAREN statement
        					| IF LPAREN expression RPAREN statement ELSE statement
        """
        if len(p)==6:
            p[0] = If(p[3],p[5],None,coord=self._token_coord(p, 1))
        else:
            p[0] = If(p[3],p[5],p[7],coord=self._token_coord(p, 1))

    def p_iteration_statement(self, p):
        """
        iteration_statement : WHILE LPAREN expression RPAREN statement
                            | FOR LPAREN expression_opt SEMI expression_opt SEMI expression_opt RPAREN statement
                            | FOR LPAREN declaration expression_opt SEMI expression_opt RPAREN statement
        """
        if len(p)==6: # while
            p[0] = While(p[3],p[5],coord=self._token_coord(p, 1))
        elif len(p)==10:
            p[0] = For(p[3],p[5],p[7],p[9],coord=self._token_coord(p, 1))
        else: # for
            p[0] = For(DeclList(decls=p[3],coord=self._token_coord(p, 1)),p[4],p[6],p[8],coord=self._token_coord(p, 1))

    def p_jump_statement(self, p):
        """
        jump_statement : BREAK SEMI
                       | RETURN expression_opt SEMI
        """
        if len(p)==3:
            p[0] = Break(coord=self._token_coord(p, 1))
        else:
            p[0] = Return(p[2],coord=self._token_coord(p, 1))

    def p_assert_statement(self, p):
        """
        assert_statement : ASSERT expression SEMI
        """
        p[0] = Assert(p[2],coord=self._token_coord(p, 1))

    def p_print_statement(self, p):
        """
        print_statement : PRINT LPAREN expression RPAREN SEMI
                        | PRINT LPAREN RPAREN SEMI
        """
        if len(p)==6:
            p[0] = Print(p[3], coord=self._token_coord(p, 1))
        else:
            p[0] = Print(None, coord=self._token_coord(p, 1))

    def p_read_statement(self, p):
        """
        read_statement : READ LPAREN expression RPAREN SEMI
        """
        p[0] = Read(p[3],coord=self._token_coord(p, 1))

    def p_empty(self, p):
        """empty : """
        p[0] = None

    def p_error(self, p):
        if p:
            print("Error near the symbol %s at line %d" % (p.value, p.lineno))
        else:
            print("Error at the end of input")

    precedence = (
    ('left', 'OR'),
    ('left', 'AND'),
    ('left', 'EQ', 'NE'),
    ('left', 'LT', 'LE', 'GT', 'GE'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE', 'MOD'),
    ('left', 'ADDRESS', 'NOT'),
    )

if __name__ == '__main__':

    import sys

    p = Parser()
    code = open(sys.argv[1]).read()
    ast = p.parse(code)
    ast.show()
    #print(ast)
