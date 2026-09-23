"""
Benchmark representation module for program synthesis.
"""

from benchmark.benchmark import Benchmark


class PSBenchmark(Benchmark):
    """
    Represents a program synthesis benchmark that is
    represented with examples/counterexamples.
    """

    generator: callable
    examples: list
    args: list
    dataset: list

    def __init__(self, generator_: callable, args_: list):
        self.generator = generator_
        self.args = args_
        self.generate()

    def generate(self):
        self.dataset = self.generator(*self.args)

    @staticmethod
    def generate_counterexamples(examples: list, n: int):
        """
        Calculates the counter examples based on the set of (positive) examples.

        :param n: Number of counter examples to be generated
        :return list of counterexamples
        """
        counterexamples = [i for i in range(n) if n not in examples]
        return counterexamples

    @staticmethod
    def generate_dataset(examples: list, n: int):
        """
        Generates a labeled dataset of n examples.

        :param examples: List with examples‚
        :param n: Number of examples
        :return dataset
        """
        dataset = []
        for i in range(n):
            if i in examples:
                dataset.append((i, 1))
            else:
                dataset.append((i, 0))
        return dataset
