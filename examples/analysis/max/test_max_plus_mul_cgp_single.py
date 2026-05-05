"""
Run script to apply CGP to the MAX problem.

This is a single-instance run script that performs one instances for a predefined setting of D.

The parameters for MAX, T and D, are passed to script via argv.
"""

import sys
from src.analysis.models.simple_cgp import SimpleCGP, SimpleCGPConfig, MutationType
from src.analysis.benchmarks.max.max import MaxPlusMul
from src.gp.tiny_cgp import *
from src.gp.functions import ADD, MUL
from src.gp.tinyverse import Const

MAX_GENERATIONS = 2000000
MAX_TIME = 9999999
D = int(sys.argv[1])
T = int(sys.argv[2])
MAX_ARITY = 2
NUM_GENES = (MAX_ARITY + 1) * D  + 1
MUTATION_RATE = 1 / NUM_GENES
functions = [ADD, MUL]
terminals = [Const(T)]

config = SimpleCGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=None,
    minimizing_fitness=False,
    ideal_fitness=None,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=1,
    num_outputs=1,
    report_interval=1,
    max_time=MAX_TIME,
    mutation_type=MutationType.PROB,
    global_seed=None,
    checkpoint_interval=9999999,
    checkpoint_dir='../checkpoint',
    experiment_name='max_tgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=1,
    population_size=2,
    num_function_nodes=D,
    levels_back=D,
    mutation_rate=MUTATION_RATE,
    strict_selection=False,
)


problem = MaxPlusMul(d=D, t=T)
config.ideal_fitness = problem.ideal
config.global_seed = int(time.time_ns())
cgp = SimpleCGP(functions, terminals, config, hyperparameters)
cgp.evolve(problem)
print(f"{D},simple_cgp,{cgp.generation_number}")
