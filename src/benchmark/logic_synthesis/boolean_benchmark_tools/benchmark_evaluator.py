"""
Provides an evaluator that can be used for the evalution of the benchmarks of the
General Boolean Function Benchmark Suite (GBFS):
https://dl.acm.org/doi/abs/10.1145/3594805.3607131
"""


class BenchmarkEvaluator:
    """Evaluator class that evaluates compressed and uncompressed output pairs
    by calculating the the hamming distance."""

    def evaluate(
        self, x: list, y: list, compressed: bool = False, bit_length: int = None
    ) -> int:
        """
        Triggers the experiments with the Hamming distance. Distinguishes between compressed and uncompressed
        output pairs of a candidate program/model and the real output values of a truth table that are passed to the function.
        """
        if len(x) != len(y):
            raise ValueError("Dimensions do not match.")

        if not compressed:
            return self.hamming_distance(x, y)
        else:
            return self.hamming_distance(
                x, y, compressed=compressed, bit_length=bit_length
            )

    def hamming_distance(
        self, x: list, y: list, compressed: bool = False, bit_length: int = None
    ) -> int:
        """
        Calculates the hamming distance which is defined as
        the number of different bits between x and y.

        :param x: list with output values from the candidate model or program
        :param y: list with real output values
        :compressed: status of truthtable (compressed or uncompressed)
        :bit_length: number of bits for each chunk of an uncompressed truth table
        """
        dist = 0
        if compressed:
            for xi, yi in zip(x, y):
                # Bitwise XOR the chunks to identify dissimilar bits
                cmp = xi ^ yi
                for i in range(bit_length):
                    # Sum up the number of 1s then
                    dist += cmp & 1
                    cmp = cmp >> 1
        else:
            for xi, yi in zip(x, y):
                if xi != yi:
                    dist += 1
        return dist
