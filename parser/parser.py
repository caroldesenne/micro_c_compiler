import ply.yacc as yacc
# Get the token map from the lexer.
from calclex import tokens

class Parser():

    def p_program(self, p):
        """ program  : global_declaration_list"""
        p[0] = p[1]

    def p_global_declaration_list(self, p):
        """ 
        global_declaration_list : global_declaration
                                | global_declaration_list global_declaration
        """
        if len(p)==2:
            p[0] = [p[1]]
        else:
            p[0] = p[1]+[p[2]]

    def p_global_declaration(self, p):
        """
        global_declaration : function_definition
                           | declaration
        """
        p[0] = p[1]

    def p_function_definition(self, p):
        """
        ???
        """
        # TODO

    def p_type_specifier(self, p):
        """
        type_specifier : VOID
                       | CHAR
                       | INT
                       | FLOAT
        """
        p[0] = p[1]

    def p_declarator(self, p):
        """
        declarator : identifier
                   | LPAREN declarator RPAREN
                   | declarator LBRACKET {constant_expression}? RBRACKET
                   | declarator LPAREN parameter_list RPAREN
                   | declarator LPAREN {identifier}* RPAREN
        """
        # TODO

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
        p[0] = (p[2], p[1], p[3])

    def p_cast_expression(self, p):
        """
        cast_expression : unary_expression
                        | LPAREN type_specifier RPAREN cast_expression
        """
        if len(p)==2:
            p[0] = p[1]
        else:
            p[0] = ('cast',p[1],p[2])

    def p_unary_expression(self, p):
        """
        unary_expression : postfix_expression
                         | PLUSPLUS unary_expression
                         | MINUSMINUS unary_expression
                         | unary_expression cast_expression
        """
        if len(p)==2:
            p[0] = p[1]
        elif p[1]=='++':
            p[0] = ('++',p[2])
        elif p[1]=='++':
            p[0] = ('--',p[2])
        else:
            p[0] = (p[1],p[2])

    def p_postfix_expression(self, p):
        """
        postfix_expression : primary_expression
                           | postfix_expression LBRACKET expression RBRACKET
                           | postfix_expression LPAREN {<assignment_expression>}* RPAREN
                           | postfix_expression PLUSPLUS
                           | postfix_expression MINUSMINUS
        """
        # TODO

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
        if len(p)==2:
            p[0] = p[1]
        else:
            p[0] = (',',p[1],p[3])

    def p_assignment_expression(self, p):
        """
        assignment_expression : binary_expression
                              | unary_expression assignment_operator assignment_expression
        """
        if len(p)==2:
            p[0] = p[1]
        else:
            p[0] = (p[2],p[1],p[3])

    def p_assignment_operator(self, p):
        """
        assignment_operator : EQUALS
                            | TIMESEQUAL
                            | DIVEQUAL
                            | MODEQUAL
                            | PLUEQUAL
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
        if len(p)==2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_parameter_declaration(self, p):
        """
        parameter_declaration : {<type_specifier>} declarator
        """
        p[0] = (p[1],p[2])
        #TODO

    def p_declaration(self, p):
        """
        declaration : {type_specifier} {init_declarator}* SEMI
        """
        #TODO

    def p_init_declarator(self, p):
        """
        init_declarator : declarator
                        | declarator EQUALS initializer
        """
        if len(p)==2:
            p[0] = p[1]
        else:
            p[0] = ('=',p[1],p[3])

    def p_initializer(self, p):
        """
        initializer : assignment_expression
                    | { initializer_list }
                    | { initializer_list , }
        """
        #TODO

    def p_initializer_list(self, p):
        """
        initializer_list : initializer
                         | initializer_list COMMA initializer
        """
        if len(p)==2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_compound_statement(self, p):
        """
        { {<declaration>}* {<statement>}* }
        """
        #TODO

    def p_statement(self, p):
        """
        statement : expression_statement
                  | compound_statement
                  | selection_statement
                  | iteration_statement
                  | jump_statement
                  | assert_statement
                  | print_statement
                  | read_statement
        """
        p[0] = p[1]

    def p_expression_statement(self, p):
        """
        expression_statement : {<expression>}? SEMI
        """
        #TODO

    def p_selection_statement(self, p):
        """
        selection_statement : IF LPAREN expression RPAREN statement
                            | IF LPAREN expression RPAREN statement ELSE statement

        """
        #TODO

    def p_iteration_statement(self, p):
        """
        iteration_statement : WHILE LPAREN expression RPAREN statement
                            | FOR LPAREN {expression}? SEMI {expression}? SEMI {expression}? RPAREN statement
        """
        #TODO

    def p_jump_statement(self, p):
        """
        jump_statement : BREAK SEMI
                       | RETURN {expression}? SEMI
        """
        #TODO

    def p_assert_statement(self, p):
        """
        assert_statement : ASSERT expression SEMI
        """
        p[0] = ('assert',p[2])

    def p_print_statement(self, p):
        """
        print_statement : PRINT LPAREN <expression>* RPAREN SEMI
        """
        #TODO

    def p_read_statement(self, p):
        """
        read_statement : READ LPAREN <declarator>+ RPAREN SEMI
        """
        #TODO
    
    def p_error (p):
        if p:
            print("Error near the symbol %s" % p.value)
        else:
            print("Error at the end of input")

    precedence = (
    ('left', 'PLUS'),
    ('left', 'TIMES')
    ) #TODO are there others?

    parser = yacc.yacc(write_tables=False)
