import math
import random
import numpy as np
from enum import Enum
from src.benchmark.benchmark import Benchmark

class DatasetType(Enum):
    UNIFORM = 0
    GRID = 1

class FunctionType(Enum):
    POLYNOMIAL = 0
    MONOMIAL = 1


class TinySR(Benchmark):
    """
    TinySR provides a minimalistic benchmark for symbolic regression intended to be used for runtime analysis.
    This class compiles an incremental set of univariate polynomial functions.
    """

    x_min: int
    x_max: int
    deg_min: int
    deg_max: int
    n: int
    step: int = 1
    fun_type = FunctionType
    ds_type: DatasetType

    def __init__(self, x_min: float, x_max: float, deg_min: int, deg_max: int, n: int,
                 fun_type: FunctionType = fun_type.POLYNOMIAL, ds_type:DatasetType=DatasetType.UNIFORM, step: int=1):

        assert deg_min < deg_max
        assert x_min < x_max

        self.x_min = x_min
        self.x_max = x_max
        self.deg_min = deg_min
        self.deg_max = deg_max
        self.n = n
        self.ds_type = ds_type
        self.fun_type = fun_type
        self.step = step

    def polynomial(self, x: float, deg: int) -> float:
        s = 0.0
        for d in range(1, deg + 1):
            s += x ** d
        return s

    def monomial(self, x: float, deg: int) -> float:
        return x ** deg

    def evenly_spaced_grid(self, start: float, stop: float, step: int) -> list[list[float]]:
        n = math.floor((abs(start) + abs(stop)) / step)
        grid = np.linspace(start, stop, n)
        return [[x] for x in grid]


    def uniform_set(
            self, a: float, b: float, n: int
    ) -> list[list[float]]:
        """
        Samples a data net of n datapoints drawn from uniform distribution in the
        closed interval [a, b]

        :param a: minimum
        :param b: maximum
        :param n: number of samples
        :return: set of sampled values
        """
        sample = []
        for _ in range(n):
            sample.append([random.uniform(a, b)])
        return sample

    def generate_dataset(self, deg: int) -> tuple[list[float], float]:
        if self.ds_type == DatasetType.GRID:
            X = self.evenly_spaced_grid(self.x_min, self.x, self.step)

        if self.fun_type == FunctionType.POLYNOMIAL:
            func = self.polynomial
        else:
            func = self.monomial

        if self.ds_type == DatasetType.UNIFORM:
            X = self.uniform_set(self.x_min, self.x_max, self.n)
        y = [func(x[0], deg) for x in X]
        return X,y

    def generate(self) -> dict[int, tuple[list[float], float]]:
        ds = {}
        for d in range(self.deg_min, self.deg_max + 1):
            ds[d] = self.generate_dataset(d)
        return ds