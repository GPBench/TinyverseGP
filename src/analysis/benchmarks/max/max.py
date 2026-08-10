"""
Implements the MAX problem which is one of the problems that is commonly used
for analysis of GP.

The aim of MAX is to find a program that returns the largest possible as output
given a maximum depth D and a constant (terminal) value t.

The maximum output is the ideal fitness of the formulated GP search problem.

There are two version of MAX that are implemented in this problem class:
    - max-{+}-D-t :
        MAX with maximum depth D, constant t and function set F = {+} (MaxPlus)

    - max-{+,*}-D-t :
        MAX with maximum depth D, constant t and function set F = {+, *} (MaxPlusMul)

This implementation of MAX supports log scaling to enable evaluation of MAX on large settings
of the depth D.
"""

from math import log2, pow, exp2
from sklearn.externals.array_api_extra.testing import override
from src.gp.problem import Problem
from src.gp.tinyverse import GPModel


class Max(Problem):
    log_transform: bool
    d: int

    def __init__(self, d_: int, t: int, log_scaling_: bool = False):
        assert (t > 0)
        self.d = d_
        self.log_transform = log_scaling_

    def evaluate(self, genome: any, model: GPModel, scale_log_: bool = False) -> int:
        return model.predict(genome=genome, observation=None)[0]

    @override
    def is_ideal(self, fitness: float) -> bool:
        return super().is_ideal(fitness)


class MaxPlus(Max):
    """
    Variant of MAX that only considers addition in the function set.

    Ideal value is calculated accordingly und used to check if the
    correct solution has been obtained.
    """

    def __init__(self, d: int, t: int):
        super().__init__(d_=d, t=t, log_scaling_=False)
        self.ideal = t * pow(2, d)
        self.minimizing = False


class MaxPlusMul(Max):
    """
    Variant of MAX that considers both, addition and multiplication in the function set.

    Ideal value is calculated accordingly und used to check if the
    correct solution has been obtained.
    """

    def __init__(self, d: int, t: int, log_scaling: bool = False):
        super().__init__(d_=d, t=t, log_scaling_=log_scaling)

        # Maximum output value of the program depends on whether log-scaling
        # is enabled or not
        if log_scaling:
            self.ideal = exp2(d - 1) * log2(max(2 * t, t * t))
        else:
            self.ideal = pow(max(2 * t, t * t), pow(2, d - 1))
        self.minimizing = False
