"""
Benchmark representation module for logic synthesis.
"""
from enum import Enum
from src.benchmark.benchmark import Benchmark
import src.benchmark.logic_synthesis.boolean_benchmark_tools.benchmark_reader as BenchmarkReader

class FSType(Enum):
    reduced = 1,
    extended = 2,

class LSBenchmark(Benchmark):
    """
    Class for representing logic synthesis benchmarks. Receives
    an generator function that is used to create the dataset.
    """

    benchmark: BenchmarkReader.Benchmark
    reader: BenchmarkReader.BenchmarkReader
    file: str
    name: str
    fstype: FSType

    def __init__(self, file_: str, name_ = "", fstype_ = None):
        self.file = file_
        self.name = name_
        self.fstype = fstype_
        self.reader = BenchmarkReader.BenchmarkReader()
        self.generate()

    def generate(self):
        """
        Calls the generator function and triggers the generations of the dataset.
        """
        if self.reader.file_format(self.file) == self.reader.PLU:
            self.reader.read_plu_file(self.file)
        elif self.reader.file_format(self.file) == self.reader.TT:
            self.reader.read_tt_file(self.file)

        self.benchmark = self.reader.benchmark

    def get_truth_table(self):
        return self.benchmark.table

    def __str__(self):
        print(self.name)
