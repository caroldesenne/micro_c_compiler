import ply.yacc as yacc
# Get the token map from the lexer.
from calclex import tokens

class Parser():
    
    def p_empty(p):
        """
        empty:
        """
        p[0] = None

    def p_program(self, p):
        """ 
        program : global_declaration_list
        """
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

    def p_declaration_list(self, p):
        """
        declaration_list : declaration
                         | delcaration_list declaration
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
        function_definition : type_specifier declarator declaration_list_opt compound-statement
                            | declarator declaration_list_opt compound-statement
        """
        if len(p)==5:
            p[0] = (p[1],p[2],p[3],p[4])
        else:
            p[0] = (p[1],p[2],p[3])        

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
                   | declarator LBRACKET constant_expression RBRACKET

        """
        if len(p)==2:
            p[0] = p[1]
        elif len(p)==4:
            p[0] = p[2]
        elif len(p)==5:
            p[0] = p[1]+p[2]+p[3]+p[4] 

    def p_declarator2(self, p):
        """
        declarator : declarator LBRACKET RBRACKET
                   | declarator LPAREN parameter_list RPAREN
        """
        if len(p)==4:
            p[0] = p[1]+p[2]+p[3]
        elif len(p)==5:
            p[0] = (p[1],p[3])

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
        identifier_list_opt : empty
                            | identifier_list
        """
        p[0] = p[1]

    def p_declarator3(self, p):
        """
        declarator : declarator LPAREN identifier_list_opt RPAREN
        """
        p[0] = (p[1],p[3])

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
        else:
            p[0] = (p[1],p[2])

    def p_postfix_expression(self, p):
        """
        postfix_expression : primary_expression
                           | postfix_expression PLUSPLUS
                           | postfix_expression MINUSMINUS
                           | postfix_expression LBRACKET expression RBRACKET
        """
        if len(p)==2:
            p[0] = p[1]
        if len(p)==3:
            p[0] = (p[1],p[2])
        else:
            p[0] = (p[1],p[3])

    def p_assignment_expression_list(self, p):
        """
        assignment_expression_list : assignment_expression
                                   | assignment_expression_list assignment_expression
        """
        if len(p)==2:
            p[0] = [p[1]]
        else:
            p[0] = p[1]+[p[2]]
    
    def p_assignment_expression_list_opt(self, p):
        """
        assignment_expression_list_opt : empty
                                       | assignment_expression_list
        """
        p[0] = p[1]

    def p_postfix_expression2(self, p):
        """
        postfix_expression : postfix_expression LPAREN assignment_expression_list_opt RPAREN
        """
        p[0] = (p[1],p[3])

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
            p[0] = (p[2],p[1],p[3])

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
        parameter_declaration : type_specifier declarator
        """
        p[0] = (p[1],p[2])

    def p_init_declarator_list(self, p):
        """
        init_declarator_list : init_declarator
                             | init_declarator_list init_declarator
        """
        if len(p)==2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_init_declarator_list_opt(self, p):
        """
        init_declarator_list_opt : empty
                                 | init_declarator_list
        """
        p[0] = p[1]

    def p_declaration(self, p):
        """
        declaration : type_specifier init_declarator_list_opt SEMI
        """
        p[0] = (p[1],p[2])

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
                    | LBRACE initializer_list RBRACE
                    | LBRACE initializer_list COMMA RBRACE
        """
        if len(p)==2:
            p[0] = p[1]
        else:
            p[0] = p[2]

    def p_initializer_list(self, p):
        """
        initializer_list : initializer
                         | initializer_list COMMA initializer
        """
        if len(p)==2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_statement_list(self, p):
        """
        statement_list : statement
                       | statement_list statement
        """
        if len(p)==2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_statement_list_opt(self, p):
        """
        statement_list_opt : empty
                           | statement_list
        """

    def p_compound_statement(self, p):
        """
        compound_statement : LBRACE declaration_list_opt statement_list_opt RBRACE
        """
        p[0] = (p[2],p[3])

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

    def p_expression_opt(self, p):
        """
        expression_opt : expression
                       | empty
        """

    def p_expression_statement(self, p):
        """
        expression_statement : expression_opt SEMI
        """
        p[0] = p[1]

    def p_selection_statement(self, p):
        """
        selection_statement : IF LPAREN expression RPAREN statement
                            | IF LPAREN expression RPAREN statement ELSE statement

        """
        if len(p)==6:
            p[0] = (p[1],p[3],p[5])
        else:
            p[0] = (p[1],p[3],p[5],p[6],p[7])

    def p_iteration_statement(self, p):
        """
        iteration_statement : WHILE LPAREN expression RPAREN statement
                            | FOR LPAREN expression_opt SEMI expression_opt SEMI expression_opt RPAREN statement
        """
        if len(p)==6: # while
            p[0] = (p[1],p[3],p[5])
        else: # for
            p[0] = (p[1],p[3],p[5],p[7],p[9])

    def p_jump_statement(self, p):
        """
        jump_statement : BREAK SEMI
                       | RETURN expression_opt SEMI
        """
        if len(p)==3:
            p[0] = p[1]
        else:
            p[0] = (p[1],p[2])

    def p_assert_statement(self, p):
        """
        assert_statement : ASSERT expression SEMI
        """
        p[0] = ('assert',p[2])

    def p_expression_list(self, p):
        """
        expression_list : expression
                        | expression_list expression
        """
        if len(p)==2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_print_statement(self, p):
        """
        print_statement : PRINT LPAREN expression_list RPAREN SEMI
                        | PRINT LPAREN RPAREN SEMI
        """
        if len(p)==6:
            p[0] = (p[1],p[3])
        else:
            p[0] = p[1]

    def p_declarator_list(self, p):
        """
        declarator_list : declarator
                        | declarator_list declarator
        """
        if len(p)==2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_read_statement(self, p):
        """
        read_statement : READ LPAREN declarator_list RPAREN SEMI
        """
        p[0] = (p[1],p[3])
    
    def p_error(p):
        if p:
            print("Error near the symbol %s" % p.value)
        else:
            print("Error at the end of input")

    precedence = (
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE', 'MOD'),
    )

    parser = yacc.yacc(write_tables=False)
