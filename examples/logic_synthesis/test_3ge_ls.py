
"""
Example module to test Tree GE with logic synthesis problems.

Evolves a symbolic expression for a boolean function that
is presented as a compressed or uncompressed truth table.

The logic synthesis benchmark files are located in the data folder.

The provided benchmarks are taken from the General Boolean Function Benchmark Suite (GFBS):

https://dl.acm.org/doi/10.1145/3594805.3607131
"""
from src.gp.tinyverse import GPConfig
from src.benchmark.logic_synthesis.ls_benchmark import LSBenchmark
from src.gp.tiny_ge import Tiny3GE
from src.gp.functions import AND, OR, NAND, NOR, NOT
from src.gp.tiny_3ge import TreeGEHyperparameters
from src.gp.loss import hamming_distance_bitwise
from src.gp.problem import BlackBox

benchmark = LSBenchmark('data/logic_synthesis/plu/add3.plu')
benchmark.generate()
truth_table = benchmark.get_truth_table()
num_inputs = benchmark.benchmark.num_inputs
num_outputs = benchmark.benchmark.num_outputs


config = GPConfig(
    num_jobs=1,
    max_generations=100,
    stopping_criteria=1e-6,
    minimizing_fitness=True,  # this should be used from the problem instance
    ideal_fitness=1e-6,  # this should be used from the problem instance
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=num_outputs,
    report_interval=1,
    max_time=60,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='logic_ge'
)

hyperparameters = TreeGEHyperparameters(
    pop_size=100,
    codon_size=1000,
    min_depth=4,
    max_depth=8,
    cx_rate=0.9,
    mutation_rate=0.1,
    tournament_size=2,
    penalty_value=99999
)

loss = hamming_distance_bitwise
data = truth_table.inputs
actual = truth_table.outputs
problem = BlackBox(data, actual, loss, 0, True)

functions = [AND, OR, NAND, NOR, NOT]
arguments = ['x']
grammar = {
    "<expr>": [
        "AND(<expr>, <expr>)",
        "OR(<expr>, <expr>)",
        "NAND(<expr>, <expr>)",
        "NOR(<expr>, <expr>)",
        "NOT(<expr>)",
        "<d>",
        "x",
    ],
    "<d>": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
}

tree_ge = Tiny3GE(functions, grammar, arguments, config, hyperparameters)

tree_ge.evolve(problem)

#tree_ge.print_individual_tree(tree_ge.best_individual.genome)
#tree_ge.print_individual(tree_ge.best_individual)
