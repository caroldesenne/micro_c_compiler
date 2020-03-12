import ply.lex as lex

class Lexer():
    
    def __init__(self, error_func):
        self.error_func = error_func
        self.filename = ''
        self.last_token = None

    def build(self, **kwargs):
        self.lexer = lex.lex(object=self, **kwargs)

    def reset_lineno(self):
        self.lexer.lineno = 1

    def input(self, text):
        self.lexer.input(text)

    def token(self):
        self.last_token = self.lexer.token()
        return self.last_token

    def find_tok_column(self, token):
        last_cr = self.lexer.lexdata.rfind('\n', 0, token.lexpos)
        return token.lexpos - last_cr

    # Internal auxiliary methods
    def _error(self, msg, token):
        location = self._make_tok_location(token)
        self.error_func(msg, location[0], location[1])
        self.lexer.skip(1)

    def _make_tok_location(self, token):
        return (token.lineno, self.find_tok_column(token))

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

        # assignments
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
    def t_ID(self, t):
        r'[a-zA-Z_][0-9a-zA-Z_]*'
        t.type = self.keyword_map.get(t.value, "ID")
        return t

    # regex for constants
    t_FLOAT_CONST = r'([0-9]*\.[0-9]+)|([0-9]+\.)'
    t_CHAR_CONST = r'\'.*\''
    t_INT_CONST = r'[0-9]+'

    # regex for string literals
    t_STRING_LITERAL = r'\".*?\"'

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
    t_EQ = r'=='
    t_NE = r'!='

    # regex for assignements
    t_EQUALS = r'='
    t_TIMESEQUAL = r'\*='
    t_DIVEQUAL = r'/='
    t_MODEQUAL = r'%='
    t_PLUSEQUAL = r'\+='
    t_MINUSEQUAL = r'\-='

    # regex for increment/decrement
    t_PLUSPLUS = r'\+\+'
    t_MINUSMINUS = r'\-\-'

    # regex for delimiters
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'
    t_LBRACE = r'{'
    t_RBRACE = r'}'
    t_COMMA = r','
    t_SEMI = r';'

    # Newlines
    def t_NEWLINE(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")

    t_ignore = ' \t'

    def t_comment(self, t):
        r'/\*(.|\n)*?\*/'
        t.lexer.lineno += t.value.count('\n')
        pass

    def t_error(self, t):
        msg = "Illegal character %s" % repr(t.value[0])
        self._error(msg, t)

    # Scanner (used only for test)
    def scan(self, data):
        self.lexer.input(data)
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            print(tok)

if __name__ == '__main__':

    import sys

    def print_error(msg, x, y):
        print("Lexical error: %s at %d:%d" % (msg, x, y))

    m = Lexer(print_error)
    m.build()  # Build the lexer
    m.scan(open(sys.argv[1]).read())  # print tokens

# CHECAR:
# t_CHAR_CONST


