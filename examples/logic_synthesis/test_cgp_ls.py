"""
Example module to test TGP with logic synthesis problems.

Evolves a symbolic expression for a boolean function that
is presented as a compressed or uncompressed truth table.

The logic synthesis benchmark files are located in the data folder.

The provided benchmarks are taken from the General Boolean Function Benchmark Suite (GFBS):

https://dl.acm.org/doi/10.1145/3594805.3607131
"""

from src.benchmark.logic_synthesis.ls_benchmark import LSBenchmark
from src.gp.tiny_cgp import *
from src.gp.problem import BlackBox
from src.gp.functions import *
from src.gp.loss import *
from src.gp.tinyverse import Var

benchmark = LSBenchmark('../../data/logic_synthesis/plu/add3.plu')
benchmark.generate()
truth_table = benchmark.get_truth_table()
num_inputs = benchmark.benchmark.num_inputs
num_outputs = benchmark.benchmark.num_outputs

functions = [AND, OR, NAND, NOR]
terminals = [Var(i) for i in range(num_inputs)]

config = CGPConfig(
    num_jobs=1,
    max_generations=2000000,
    stopping_criteria=0,
    minimizing_fitness=True,
    ideal_fitness=0,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=len(terminals),
    num_outputs=num_outputs,
    report_interval=5000,
    max_time=360,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='logic_cgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=1,
    population_size=1000,
    levels_back=1000,
    mutation_rate=0.1,
    num_function_nodes=30,
    strict_selection=False,
)

data = truth_table.inputs
actual = truth_table.outputs
loss = hamming_distance_bitwise
problem = BlackBox(data, actual, loss, 0, True)

cgp = TinyCGP(functions, terminals, config, hyperparameters)
cgp.evolve(problem)
