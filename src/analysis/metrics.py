import math
import numpy as np


def success_rate(iterations: list[int], i:int) -> float:
    """
    Calculates the success rate based on a set of iteration numbers up to a given iteration number i.
    """
    n_total = len(iterations)
    ns_i = len([r for r in iterations if r < i])
    return ns_i/n_total

def computational_effort(iterations: list[int], i:int, m:int, z:float) -> float:
    """
    Calculates Koza's computational effort used to measure the performance
    of a GP system.
    """
    p_i = max(min(success_rate(iterations, i), z),1.0-z)
    r_z = math.ceil(math.log(1.0 - z)/math.log(1.0-p_i))
    return m*i*r_z

def min_computational_effort(iterations: list[int], i_min:int, i_max:int, m:int, z:float) -> float:
    """
    Calculates the minimum computational effort in a predefined range of iteration numbers.
    """
    c = []
    for i in range(i_min, i_max):
        c.append(computational_effort(iterations, i, m, z))
    return min(c), np.argmin(c)

