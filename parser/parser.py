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
        primary_expression : ID
                           | INT_CONST
                           | FLOAT_CONST
                           | CHAR_CONST
                           | STRING_LITERAL
                           | LPAREN expression RPAREN
        """
        if len(p)==2: # (id,constant and string)
            p[0] = p[1]
        else: # (expression)
            p[0] = p[2]



