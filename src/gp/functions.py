"""
Implementation of arithmetic, logical and comparative functions
that are commonly used to curate the GP function set.
"""

from src.gp.tinyverse import Function
import operator
import numpy as np

def f2b(input: float) -> bool:
    """
    Converts a float to a boolean.
    """
    return True if input > 0 else False

def b2f(input: bool) -> float:
    """
    Converts a boolean to a float.
    """
    return 1.0 if input else -1.0

def p2i(x, clip=1e100):
    """
    Protected to integer conversion method.
    Checks for infinity values and converts nan values
    to to valid integer numbers.
    """
    if np.isnan(x) or np.isinf(x) or np.isneginf(x):
        x = np.nan_to_num(x)
    if not isinstance(x, int):
        np.clip(x, -clip, clip)
        return int(x)
    return x

def pdiv(x, y, eps=1e-8):
    return np.divide(x, y + (np.abs(y) < eps) * eps)

def square(x, clip=1e100):
    return np.clip(np.square(x), -clip, clip)

def cube(x, clip=1e100):
    return np.clip(np.power(x, 3), -clip, clip)

def pow(x, y, clip=1e100):
    return np.clip(np.pow(x,y), -clip, clip)

def inv(x):
    return pdiv(1.0,x)

def plog(x):
    if x == 0.0:
        return 0.0
    return np.log(np.abs(x))

def psqrt(x):
    return np.sqrt(np.abs(x))

def pexp(x, clip=50.0):
    return np.clip(np.exp(x), -clip, clip)



# Mathematical/Arithmetic
ADD = Function(2, "ADD", operator.add)
SUB = Function(2, "SUB", operator.sub)
MUL = Function(2, "MUL", operator.mul)
DIV = Function(2, "DIV", pdiv)
INV = Function(1, "INV", inv)
ABS =  Function(1, "ABS", np.abs)
EXP = Function(1, "EXP", pexp)
LOG = Function(1, "LOG", plog)
SQRT = Function(1, "SQRT", psqrt)
SQR = Function(1, "SQR", square)
CUBE = Function(1, "CUBE", cube)
POW = Function(2, "POW", pow)
SIN = Function(1, "SIN", np.sin)
COS = Function(1, "COS", np.cos)
TAN = Function(1, "TAN", np.tan)
ARCSIN = Function(1, "ARCSIN", np.arcsin)
ARCCOS = Function(1, "ARCCOS", np.arccos)
ARCTAN = Function(1, "ARCTAN", np.arctan)
ID = Function(1, "ID", lambda x: x)
CEIL = Function(1, "CEIL", np.ceil)
FLOOR = Function(1, "FLOOR", np.floor)
MOD = Function(2, "MOD", lambda x, y: x % y)
NEG = Function(1, "NEG", lambda x: -x)

# Logical/Bitwise
AND = Function(2, "AND", lambda x, y: p2i(x) & p2i(y))
OR = Function(2, "OR", lambda x, y: p2i(x) | p2i(y))
NOT = Function(1, "NOT", lambda x: ~p2i(x))
NOTA = Function(2, "NOTa", lambda x,y: ~p2i(x))
NOTB = Function(2, "NOTb", lambda x,y: ~p2i(y))
NAND = Function(2, "NAND", lambda x, y: ~(p2i(x) & p2i(y)))
NOR = Function(2, "NOR", lambda x, y: ~(p2i(x) | p2i(y)))
XOR = Function(2, "XOR", lambda x, y: p2i(x) ^ p2i(y))
XNOR = Function(2, "XNOR", lambda x, y: ~(p2i(x) ^ p2i(y)))
BUFA = Function(2, "BUFa", lambda x, y: p2i(x))
BUFB = Function(2, "BUFb", lambda x, y: p2i(x))
SHFTL = Function(1, "SHFTL", lambda x: p2i(x) << 1)
SHFTR = Function(1, "SHFTR", lambda x: p2i(x) >> 1)

# Comparison
LT = Function(2, 'LT', lambda x,y : b2f(x < y))
LTE = Function(2, 'LTE', lambda x,y : b2f(x <= y))
GT = Function(2, 'GT', lambda x,y : b2f(x > y))
GTE = Function(2, 'GTE',lambda x,y : b2f(x >= y))
EQ = Function(2, 'EQ', lambda x,y : b2f(x == y))
NEQ = Function(2, 'NEQ', lambda x,y : b2f(x != y))
MIN = Function(2, 'MIN', lambda x,y : min(x,y))
MAX = Function(2, 'MAX', lambda x,y : max(x,y))
IF = Function(3, 'IF', lambda x,y,z : y if f2b(x) else z)
IFLEZ = Function(3, 'IFLEZ', lambda x,y,z : y if x <= 0 else z)
IFGTZ = Function(3, 'IFGTZ', lambda x,y,z : y if x > 0 else z)


