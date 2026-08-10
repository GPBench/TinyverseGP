"""
Run script to apply QD-CGP to the MAX problem.

This is a single-instance run script that performs one instances for a predefined setting of D.

The parameters for MAX, T and D, are passed to script via argv.
"""

import sys
from math import log2
from src.analysis.models.simple_cgp import MutationType, SimpleCGPConfig
from src.analysis.benchmarks.max.max import MaxPlusMul
from src.analysis.benchmarks.max.log_scaling import LOG_ADD, LOG_MUL
from src.analysis.models.simple_qd_cgp import SimpleQdCGP, QdCGPHyperparameters
from src.gp.tiny_cgp import *
from src.gp.tinyverse import Const

MAX_GENERATIONS = 2000000
MAX_TIME = 9999999
D = int(sys.argv[1])
T = int(sys.argv[2])
assert(T > 1)
MAX_ARITY = 2
NUM_GENES = (MAX_ARITY + 1) * D  + 1
MUTATION_RATE = 1 / NUM_GENES
functions = [LOG_ADD, LOG_MUL]
terminals = [Const(log2(T))]

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
    mutation_type=MutationType.SAM,
    global_seed=None,
    checkpoint_interval=9999999,
    checkpoint_dir='../checkpoint',
    experiment_name='max_tgp'
)

hyperparameters = QdCGPHyperparameters(
    mu=1,
    lmbda=1,
    population_size=2,
    num_function_nodes=D,
    max_active_nodes=D,
    levels_back=D,
    discard_invalid=True,
    mutation_rate=MUTATION_RATE,
    strict_selection=False,
    cx_rate=0.5
)

if config.mutation_type == MutationType.SAM:
    appendix = "sam"
else:
    appendix = "prob"

problem = MaxPlusMul(d=D, t=T, log_scaling=True)
config.ideal_fitness = problem.ideal
config.global_seed = int(time.time_ns())
cgp = SimpleQdCGP(functions, terminals, config, hyperparameters)
cgp.evolve(problem)

print(f"{D},simple_qd_cgp_log_{appendix},{cgp.generation_number}")
