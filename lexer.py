import ply.lex as lex

class Lexer():
    
    def __init__(self):
        self.filename = ""
        self.last_token = None

    keywords = (
        'FOR','IF','PRINT', 'ASSERT', 'BREAK', 'CHAR', 'ELSE',
        'INT', 'FLOAT', 'READ', 'RETURN', 'VOID', 'WHILE',
    )

    tokens = keywords + (
        # identifiers
        'ID',

        # constants
        'INT_CONST',
        'FLOAT_CONST',
        'CHAR_CONST',

        # string literals
        'STRING_LITERAL',

        # operators
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MOD',
        'OR', 'AND', 'NOT', 'ADDRESS', 'LT', 'LE',
        'GT', 'GE', 'EQ', 'NE',

        # assignment
        'EQUALS', 'TIMESEQUAL', 'DIVEQUAL',
        'MODEQUAL', 'PLUSEQUAL', 'MINUSEQUAL',

        # increment/decrement
        'PLUSPLUS','MINUSMINUS',

        # delimiters
        'LPAREN', 'RPAREN',     # ( )
        'LBRACKET', 'RBRACKET', # [ ]
        'LBRACE', 'RBRACE',     # { }
        'COMMA', 'SEMI',        # , ;
    )

    # map for keywords
    keyword_map = {}
    for keyword in keywords:
        keyword_map[keyword.lower()] = keyword
 
    # regex for identifiers
    def t_ID(t):
        r'[a-zA-Z_][0-9a-zA-Z_]*'
        t.type = self.keyword_map.get(t.value, "ID")
	return t

    # regex for constants
    TODO

    # regex for string literals   

    # regex for operators
    t_PLUS = r'\+'
    t_MINUS = r'\-'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_MOD = r'%'
    t_OR = r'\|\|'
    t_AND = r'&&'
    t_NOT = r'!'
    t_ADDRESS = r'&'
    t_LT = r'<'
    t_LE = r'<='
    t_GT = r'>'
    t_GE = r'>='
    t_EQ = r'='
    t_NE = r'!='



