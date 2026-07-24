"""
Run script to apply QD-TGP to the MAX problem.

This is a single-instance run script that performs one instances for a predefined setting of D.

The parameters for MAX, T and D, are passed to script via argv.
"""

import sys
from math import log2
from src.analysis.benchmarks.max.max import MaxPlusMul
from src.analysis.models.simple_qd_tgp import SimpleQdTGP, QdTGPConfig, InitMethod, QdTGPHyperparameters
from src.analysis.models.simple_tgp import MutationType
from src.gp.tiny_cgp import *
from src.analysis.benchmarks.max.log_scaling import LOG_ADD, LOG_MUL
from src.gp.tinyverse import Const

MAX_GENERATIONS = 2000000
MAX_TIME = 999999
D = int(sys.argv[1])
T = int(sys.argv[2])
assert(T > 1)
MAX_DEPTH = D
functions = [LOG_ADD, LOG_MUL]
terminals = [Const(log2(T))]

config = QdTGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=None,
    minimizing_fitness=False,
    ideal_fitness=None,
    init_method=InitMethod.GROW,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1000,
    max_time=MAX_TIME,
    global_seed=None,
    checkpoint_interval=9999999,
    checkpoint_dir='../checkpoint',
    experiment_name='max_tgp'
)

hyperparameters = QdTGPHyperparameters(
    lmbda=1,
    k=1,
    strict_selection=False,
    check_size=False,
    max_depth=D,
    min_depth=1,
    multi=True,
    cx_rate=0.5,
    mutation_type=MutationType.HVL_NODE_UNBIASED
)

match config.init_method:
    case InitMethod.GROW:
        init_appendix = "grow"
    case InitMethod.FULL:
        init_appendix = "full"
    case InitMethod.MIN:
        init_appendix = "min"

match hyperparameters.mutation_type:
    case MutationType.HVL_DEPTH_UNBIASED:
        hvl_appendix1 = "depth_unbiased"
    case MutationType.HVL_NODE_UNBIASED:
        hvl_appendix1 = "node_unbiased"
    case MutationType.HVL_STD:
        hvl_appendix1 = "std"

if hyperparameters.multi:
    hvl_appendix2 = "multi"
else:
    hvl_appendix2 = "single"

problem = MaxPlusMul(d=D, t=T, log_scaling=True)
config.ideal_fitness = problem.ideal
config.global_seed = int(time.time_ns())
tgp = SimpleQdTGP(functions, terminals, config, hyperparameters)
tgp.evolve(problem)

print(f"{D},simple_qd_tgp_log_{hvl_appendix1}_{hvl_appendix2}_{init_appendix},{tgp.generation_number}")
