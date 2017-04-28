# Reverse Polish Notation
# ~Gros

from collections import namedtuple
import math

Info = namedtuple('Info', 'prec assoc args')
L, R = 'Left Right'.split()
OPERATOR, FUNC, SEPARATOR, NUM, VAR, LPAREN, RPAREN = 'OPERATOR FUNCTION SEPARATOR NUMBER VARIABLE ( )'.split()
operators = {
    '^': Info(prec=5, assoc=R, args=2),
    '*': Info(prec=3, assoc=L, args=2),
    '/': Info(prec=3, assoc=L, args=2),
    '//': Info(prec=3, assoc=L, args=2),
    '%': Info(prec=1, assoc=L, args=2),
    '+': Info(prec=2, assoc=L, args=2),
    '-': Info(prec=2, assoc=L, args=2),
    '#': Info(prec=4, assoc=L, args=1),  # unary minus
}
functions = {
    'max': Info(prec=0, assoc=L, args=2),
    'min': Info(prec=0, assoc=L, args=2),
    'acos': Info(prec=0, assoc=L, args=1),
    'acosh': Info(prec=0, assoc=L, args=1),
    'asin': Info(prec=0, assoc=L, args=1),
    'asinh': Info(prec=0, assoc=L, args=1),
    'atan': Info(prec=0, assoc=L, args=1),
    'atan2': Info(prec=0, assoc=L, args=1),
    'atanh': Info(prec=0, assoc=L, args=1),
    'ceil': Info(prec=0, assoc=L, args=1),
    'copysign': Info(prec=0, assoc=L, args=2),
    'cos': Info(prec=0, assoc=L, args=1),
    'cosh': Info(prec=0, assoc=L, args=1),
    'degrees': Info(prec=0, assoc=L, args=1),
    'erf': Info(prec=0, assoc=L, args=1),
    'erfc': Info(prec=0, assoc=L, args=1),
    'exp': Info(prec=0, assoc=L, args=1),
    'expm1': Info(prec=0, assoc=L, args=1),
    'fabs': Info(prec=0, assoc=L, args=1),
    'factorial': Info(prec=0, assoc=L, args=1),
    'floor': Info(prec=0, assoc=L, args=1),
    'fmod': Info(prec=0, assoc=L, args=2),
    'gamma': Info(prec=0, assoc=L, args=1),
    'hypot': Info(prec=0, assoc=L, args=2),
    'ldexp': Info(prec=0, assoc=L, args=2),
    'lgamma': Info(prec=0, assoc=L, args=1),
    'log': Info(prec=0, assoc=L, args=2),
    'log10': Info(prec=0, assoc=L, args=1),
    'log1p': Info(prec=0, assoc=L, args=1),
    'pow': Info(prec=0, assoc=L, args=3),
    'radians': Info(prec=0, assoc=L, args=1),
    'sin': Info(prec=0, assoc=L, args=1),
    'sinh': Info(prec=0, assoc=L, args=1),
    'sqrt': Info(prec=0, assoc=L, args=1),
    'tan': Info(prec=0, assoc=L, args=1),
    'tanh': Info(prec=0, assoc=L, args=1),
    'trunc': Info(prec=0, assoc=L, args=1),
}
constants = ('e', 'pi')


class RPNparserError(BaseException):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class RPNparser(object):
    def __init__(self, equation):
        """RPN parser

        Args:
            equation(string)
        Fields:
            parsed(list): RPN (postfix) representation of equation
        """
        self.parsed = self.shunting_yard(equation)

    @staticmethod
    def tokens_to_types_values(tokens):
        """Convert equation to tuples (TYPE, token)

        Args:
            tokens(string/list of strings): token is one of:
                operators, functions, left/right parenthesis, function separator, constant, number, alnum variable
        Returns:
            list of tuples (TYPE, value)
                TYPE in [OPERATOR, FUNC, SEPARATOR, NUM, LPAREN, RPAREN]
                value is a token
        """
        if type(tokens) != list:
            for token_type in operators.keys() + functions.keys() + list('(),'):
                tokens = tokens.replace(token_type, ' '+token_type+' ')
            tokens = tokens.split()

        types_tokens = []
        for token in tokens:
            if token in operators:
                types_tokens.append((OPERATOR, token))
            elif token == '(':
                types_tokens.append((LPAREN, token))
            elif token == ')':
                types_tokens.append((RPAREN, token))
            elif token in functions:
                types_tokens.append((FUNC, token))
            elif token == ',':
                types_tokens.append((SEPARATOR, token))
            elif token.replace('.', '').isdigit():
                types_tokens.append((NUM, token))
            elif token.replace('.', '').isalnum():
                types_tokens.append((VAR, token))
            else:
                raise RPNparserError("Found invalid token: {}".format(repr(token)))
        return types_tokens

    def shunting_yard(self, equation_string):
        """Shunting-yard algorithm, converts infix to postfix (RPN) notation

        Args:
            equation_string(string)
        Returns:
            list of tokens
        """
        types_tokens = self.tokens_to_types_values(equation_string)
        output, stack = [], []
        for position, (token_type, token) in enumerate(types_tokens):
            # ----------- NUMBERS
            if token_type in (NUM, VAR):
                output.append(token)

            # ----------- FUNCTIONS
            elif token_type == FUNC:
                stack.append(token)

            # ----------- SEPARATORS
            elif token_type == SEPARATOR:
                while stack and stack[-1] != LPAREN:
                    output.append(stack.pop())
                if len(stack) == 0 or stack[-1] != LPAREN:
                    raise RPNparserError("Separator misplace od parentheses mismatch")

            # ---------- LEFT PARENTHESIS
            elif token_type == LPAREN:
                stack.append(token)

            # ---------- RIGHT PARENTHESIS
            elif token_type == RPAREN:
                while stack and stack[-1] != LPAREN:
                    output.append(stack.pop())
                if len(stack) == 0 or stack[-1] != LPAREN:
                    raise RPNparserError("Lack of left parenthesis")
                stack.pop()
                if stack and stack[-1] in functions:
                    output.append(stack.pop())

            # ----------- OPERATOR
            elif token_type == OPERATOR:
                o1 = token
                # check if unary minus
                if o1 == '-':
                    if position == 0 or types_tokens[position-1][0] in (OPERATOR, LPAREN):
                        o1 = '#'
                while stack and stack[-1] in operators:
                    o2 = stack[-1]
                    if operators[o1].args != 1 and \
                            (operators[o1].assoc == L and operators[o1].prec <= operators[o2].prec) or \
                            (operators[o1].assoc == R and operators[o1].prec < operators[o2].prec):
                        output.append(stack.pop())
                    else:
                        break
                stack.append(o1)
            else:
                raise RPNparserError("Unrecognized token: " + repr(token_type))

        while stack:
            if stack[-1] in '()':
                raise RPNparserError("Mismatched parenthesis")
            output.append(stack.pop())
        return output

    def compute(self, **kwargs):
        types_tokens = self.tokens_to_types_values(self.parsed)
        stack = []
        for token_type, token in types_tokens:
            if token_type == NUM:
                stack.append(token)
            elif token_type == VAR:
                if token in constants:
                    token = "math."+token
                elif token in kwargs:
                    token = kwargs[token]
                else:
                    raise RPNparserError("Unknow variable: {}".format(repr(token)))
                stack.append(token)
            else:
                if token_type == OPERATOR:
                    n = operators[token].args
                elif token_type == FUNC:
                    n = functions[token].args
                else:
                    raise RPNparserError("Wrong token type: {}".format(repr(token_type)))

                if len(stack) < n:
                    raise RPNparserError("Not enough operator/function arguments")

                args = map(str, [stack.pop() for _ in xrange(n)][::-1])
                if token_type == OPERATOR:
                    if operators[token].args == 1:
                        if token == '#':
                            token = '-'
                        tmp = token + args[0]
                    elif operators[token].args == 2:
                        if token == '^':
                            token = '**'
                        tmp = args[0] + token
                        if token == '/':
                            tmp += 'float(' + args[1] + ')'
                        else:
                            tmp += args[1]
                    else:
                        raise RPNparserError("Operator with wrong number of arguments")
                else:
                    tmp = "math." + token + '(' + ', '.join(args) + ')'
                tmp = eval(tmp)
                stack.append(tmp)

        if len(stack) != 1:
            raise RPNparserError("Something is wrong after parsing to infix notation")
        return stack[0]

    def infix(self):
        token_type = self.tokens_to_types_values(self.parsed)
        stack = []
        for token_type, token in token_type:
            if token_type in (NUM, VAR):
                stack.append(token)
            else:
                if token_type == OPERATOR:
                    n = operators[token].args
                elif token_type == FUNC:
                    n = functions[token].args
                else:
                    raise RPNparserError("Wrong token type: {}".format(repr(token_type)))

                if len(stack) < n:
                    raise RPNparserError("Not enough operator/function arguments")

                args = [stack.pop() for _ in xrange(n)][::-1]
                if token_type == OPERATOR:
                    if operators[token].args == 1:
                        if token == '#':
                            token = '-'
                        tmp = '( ' + token + args[0] + ' ) '
                    elif operators[token].args == 2:
                        tmp = '( ' + args[0] + ' ' + token + ' ' + args[1] + ' ) '
                    else:
                        raise RPNparserError("Operator with wrong number of arguments")
                else:
                    tmp = token + '(' + ', '.join(args) + ') '
                stack.append(tmp)

        if len(stack) != 1:
            raise RPNparserError("Something is wrong after parsing to infix notation")
        result = stack[0].strip().strip('(').strip(')')
        result = ' '.join(result.split())
        return result


def test_rpn_manually():
    while True:
        eq = raw_input("Equation: ")  # i.e. (pi * x2) + sin(tan(5/3)) + fabs(x1)
        if eq.lower() in ('', 'end', 'quit'):
            break
        try:
            onp = RPNparser(eq)
            print "RPN: ", onp.parsed
            print "Infix: ", onp.infix()
            print "Value: ", onp.compute(x1=-1, x2=2)
            print ''
        except RPNparserError, e:
            print '\nError:', e

if __name__ == "__main__":
    test_rpn_manually()