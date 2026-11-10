"""
Benchmark representation module for symbolic regression.
"""

import random
import numpy as np
from benchmark.benchmark import Benchmark


def koza1(x):
    return pow(x, 4) + pow(x, 3) + pow(x, 2) + x

def koza2(x):
    return pow(x, 5) - 2 * pow(x, 3) + x

def koza3(x):
    return pow(x, 6) - 2 * pow(x, 4) + pow(x, 2)

class SRBenchmark(Benchmark):
    """
    Represents a symbolic regression benchmark that is based on a uniformly sampled dataset.
    """

    def dataset_uniform(
        self, a: int, b: int, n: int, dimension: int, benchmark: str
    ) -> tuple:
        """
        Samples a data net of n datapoints drawn from uniform distribution in the
        closed interval [a, b]

        :param a: minimum
        :param b: maximum
        :param n: number of samples
        :param dimension: dimension of the objective function
        :param benchmark: benchmark function
        :return: samples and corresponding actual function values
        """
        sample = []
        point = []
        for _ in range(n):
            point.clear()
            for _ in range(dimension):
                point.append(random.uniform(a, b))
            sample.append(point.copy())
        values = [self.objective(benchmark, point) for point in sample]
        return sample, values

    @staticmethod
    def random_set(min, max, n, objective, dim=1):
        def random_samples(min, max, n, dim=1):
            assert min < max
            samples = []

            for idx in range(0, dim):
                sample = (max - min) * np.random.random_sample(n) + min
                samples.append(np.array(sample, dtype=np.float32))

            return np.stack(samples, axis=1)

        samples = random_samples(min, max, n, dim)
        values = [objective(point) for point in samples]

        return samples, np.array(values, dtype=np.float32)

    def generate(self, benchmark: str) -> tuple:
        """
        Generates the dataset for a selected benchmark function.

        :param benchmark: selected benchmark function
        :return: generated dataset
        """
        match benchmark:
            case "KOZA1":
                return self.dataset_uniform(-10, 10, 20, 1, benchmark)
            case "KOZA2":
                return self.dataset_uniform(-10, 10, 20, 1, benchmark)
            case "KOZA3":
                return self.dataset_uniform(-10, 10, 20, 1, benchmark)
            case "DEBUG":
                return self.dataset_uniform(-4, 4, 100, 1, benchmark)

    def objective(self, benchmark: str, args: list) -> float:
        """
        Calculates the actual/objective value of the benchmark function.

        :param benchmark: Selected benchmark function
        :param args: list of arguments

        :return: objective function value
        """
        match benchmark:
            case "KOZA1":
                return pow(args[0], 4) + pow(args[0], 3) + pow(args[0], 2) + args[0]
            case "KOZA2":
                return pow(args[0], 5) - 2 * pow(args[0], 3) + args[0]
            case "KOZA3":
                return pow(args[0], 6) - 2 * pow(args[0], 4) + pow(args[0], 2)
            case "DEBUG":
                return pow(args[0], 3) - pow(args[0], 2) + pow(args[0], 4)
