"""
Example module to test GE with logic synthesis problems.

Evolves a symbolic expression for a boolean function that
is presented as a compressed or uncompressed truth table.

The logic synthesis benchmark files are located in the data folder.

The provided benchmarks are taken from the General Boolean Function Benchmark Suite (GFBS):

https://dl.acm.org/doi/10.1145/3594805.3607131
"""
from src.gp.tinyverse import GPConfig
from src.benchmark.logic_synthesis.ls_benchmark import LSBenchmark
from src.gp.tiny_ge import TinyGE, GEHyperparameters
from src.gp.functions import AND, OR, NAND, NOR, XOR, NOT
from src.gp.loss import hamming_distance_bitwise
from src.gp.problem import BlackBox

benchmark = LSBenchmark("data/logic_synthesis/plu/add3.plu")
benchmark.generate()
truth_table = benchmark.get_truth_table()
num_inputs = benchmark.benchmark.num_inputs
num_outputs = benchmark.benchmark.num_outputs


config = GPConfig(
    num_jobs=1,
    max_generations=500_00,
    stopping_criteria=1e-6,
    minimizing_fitness=True,  # this should be used from the problem instance
    ideal_fitness=0,  # this should be used from the problem instance
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=num_outputs,
    report_interval=100,
    max_time=500,
    global_seed=None,
    checkpoint_interval=100000,
    checkpoint_dir='examples/checkpoint',
    experiment_name='logic_ge'
)

hyperparameters = GEHyperparameters(
    pop_size=1000,
    genome_length=60,
    codon_size=100,
    cx_rate=0.95,
    mutation_rate=0.25,
    tournament_size=3,
    penalty_value=99999,
)

loss = hamming_distance_bitwise
data = truth_table.inputs
actual = truth_table.outputs
problem = BlackBox(data, actual, loss, 0, True)

functions = [AND, OR, NAND, NOR, XOR, NOT]
arguments = ["a", "b", "c", "d", "e", "f", "g"]
grammar = {
    "<expr>": ["[<lexpr>, <lexpr>, <lexpr>, <lexpr>]"],
    "<lexpr>": [
        "AND(<vexpr>, <vexpr>)",
        "OR(<vexpr>, <vexpr>)",
        "NAND(<vexpr>, <vexpr>)",
        "NOR(<vexpr>, <vexpr>)",
        "XOR(<vexpr>, <vexpr>)",
        "NOT(<vexpr>)",
    ],
    "<vexpr>": [
        "AND(<vexpr>, <vexpr>)",
        "OR(<vexpr>, <vexpr>)",
        "NAND(<vexpr>, <vexpr>)",
        "NOR(<vexpr>, <vexpr>)",
        "XOR(<vexpr>, <vexpr>)",
        "NOT(<vexpr>)",
        "<var>"
    ],
    "<var>": ["a", "b", "c", "d", "e", "f", "g"]
}


ge = TinyGE(functions, grammar, arguments, config, hyperparameters)
result = ge.evolve(problem)
print(ge.expression(result.genome))
