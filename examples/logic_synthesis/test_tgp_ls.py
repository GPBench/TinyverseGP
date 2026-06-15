"""
Example module to test TGP with logic synthesis problems.

Evolves a symbolic expression for a boolean function that
is presented as a compressed or uncompressed truth table.

The logic synthesis benchmark files are located in the data folder.

The provided benchmarks are taken from the General Boolean Function Benchmark Suite (GFBS):

https://dl.acm.org/doi/10.1145/3594805.3607131
"""

from src.benchmark.logic_synthesis.ls_benchmark import LSBenchmark
from src.gp.tiny_tgp import *
from src.gp.functions import *
from src.gp.loss import *
from src.gp.problem import BlackBox

benchmark = LSBenchmark('../../data/logic_synthesis/plu/add3.plu')
benchmark.generate()
truth_table = benchmark.get_truth_table()
num_inputs = benchmark.benchmark.num_inputs
num_outputs = benchmark.benchmark.num_outputs

functions = [AND, OR, NAND, NOR]
terminals = [Var(i) for i in range(num_inputs)]

# 38
config = GPConfig(
    num_jobs=1,
    max_generations=1000,
    stopping_criteria=1e-6,
    minimizing_fitness=True,  # this should be used from the problem instance
    ideal_fitness=1e-6,  # this should be used from the problem instance
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=num_outputs,
    report_interval=1,
    max_time=360,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='logic_tgp'
)

hyperparameters = TGPHyperparameters(
    pop_size=2000,
    max_size=25,
    max_depth=5,
    cx_rate=0.9,
    mutation_rate=0.3,
    tournament_size=2,
    erc=False
)

loss = hamming_distance_bitwise
data = truth_table.inputs
actual = truth_table.outputs
problem = BlackBox(data, actual, loss, 0, True)

tgp = TinyTGP(functions, terminals, config, hyperparameters)
result = tgp.evolve(problem)
print(tgp.expression(result.genome))
