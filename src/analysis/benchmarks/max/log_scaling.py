"""
This module contains all functions and operators needed to perform the log scaling to obtain results for large
settings of D since the program output on the non-scaled MAX problem can be huge.

For the log transform we use the log of base 2.
"""

from math import log2, exp2
from src.gp.tinyverse import Function


def transform_log(x: int):
   return 0 if x == 0 else log2(x)


def log_add(a: int, b: int, log_output: bool=False):
    """
    Performs the log addition of two variables.
    """
    if a == 0:
        res = b
    elif b == 0:
        res = a
    else:
        res = max(a, b) + log2(1 + exp2(-abs(a - b)))
    return transform_log(res) if log_output else res


def log_mul(a: int, b: int, log_output: bool=False):
    """
    Performs the log multiplication of input variables.
    """
    if a == 0 or b == 0:
        res = 0
    else:
        res = a + b
    return transform_log(res) if log_output else res


# The arithmetic log functions are used to create new operators that can be
# added to the functions set.
LOG_ADD = Function(2, "LOG_ADD", log_add)
LOG_MUL = Function(2, "LOG_MUL", log_mul)
