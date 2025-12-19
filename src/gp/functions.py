"""
Implementation of arithmetic, logical and comparative functions
that are commonly used to curate the GP function set.
"""

from src.gp.tinyverse import Function
import operator
import numpy as np

def f2b(input: float):
    """
    Converts a float to a boolean.
    """
    return True if input > 0 else False

def b2f(input: bool):
    """
    Converts a boolean to a float.
    """
    return 1.0 if input else -1.0

def pdiv(x, y, eps=1e-8):
    return np.divide(x, y + (np.abs(y) < eps) * eps)

def square(x, clip=1e100):
    return np.clip(np.square(x), -clip, clip)

def cube(x, clip=1e100):
    return np.clip(np.power(x, 3), -clip, clip)


def plog(x):
    if x == 0.0:
        return 0.0
    return np.log(np.abs(x))

def psqrt(x):
    return np.sqrt(np.abs(x))

def pexp(x, clip=50.0):
    return np.clip(np.exp(x), -clip, clip)

# Arithmetic Functions
ADD = Function(2, "Add", operator.add)
SUB = Function(2, "Sub", operator.sub)
MUL = Function(2, "Mul", operator.mul)
DIV = Function(2, "Div", pdiv)
EXP = Function(1, "EXP", np.exp)
LOG = Function(1, "LOG", plog)
SQRT = Function(1, "SQRT", psqrt)
SQR = Function(1, "SQR", square)
CUBE = Function(1, "CUBE", cube)
POWER = Function(2, "Power", operator.pow)

# Logical Functions
AND = Function(2, "AND", lambda x, y: int(x) & int(y))
OR = Function(2, "OR", lambda x, y: int(x) | int(y))
NOT = Function(1, "NOT", lambda x: ~int(x))
NOTA = Function(2, "NOTa", lambda x,y: ~int(x))
NOTB = Function(2, "NOTb", lambda x,y: ~int(y))
NAND = Function(2, "NAND", lambda x, y: ~(int(x) & int(y)))
NOR = Function(2, "NOR", lambda x, y: ~(int(x) | int(y)))
XOR = Function(2, "XOR", lambda x, y: int(x) ^ int(y))
XNOR = Function(2, "XNOR", lambda x, y: ~(int(x) ^ int(y)))
BUFA = Function(2, "BUFa", lambda x, y: x)
BUFB = Function(2, "BUFb", lambda x, y: y)

# Policy Search / Classification
LT = Function(2, 'LT', lambda x,y : b2f(x < y))
LTE = Function(2, 'LTE', lambda x,y : b2f(x <= y))
GT = Function(2, 'GT', lambda x,y : b2f(x > y))
GTE = Function(2, 'GTE',lambda x,y : b2f(x >= y))
EQ = Function(2, 'EQ', lambda x,y : b2f(x == y))
MIN = Function(2, 'MIN', lambda x,y : min(x,y))
MAX = Function(2, 'MAX', lambda x,y : max(x,y))
NEG = Function(1, 'NEQ', lambda x : operator.neg(x))
IF = Function(3, 'IF', lambda x,y,z : y if f2b(x) else z)
IFLEZ = Function(3, 'IFLEZ', lambda x,y,z : y if x <= 0 else z)
IFGTZ = Function(3, 'IFGTZ', lambda x,y,z : y if x > 0 else z)



