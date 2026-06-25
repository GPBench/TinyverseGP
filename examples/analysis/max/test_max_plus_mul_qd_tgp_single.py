"""
Run script to apply QD-TGP to the MAX problem.

This is a single-instance run script that performs one instances for a predefined setting of D.

The parameters for MAX, T and D, are passed to script via argv.
"""

import sys
from src.analysis.benchmarks.max.max import MaxPlusMul
from src.analysis.models.simple_qd_tgp import SimpleQdTGP
from src.gp.tiny_cgp import *
from src.gp.functions import ADD, MUL
from src.gp.tiny_tgp import TGPConfig
from src.gp.tinyverse import Const
from src.analysis.models.simple_tgp import SimpleTGP, SimpleTGPHyperparameters

MAX_GENERATIONS = 2000000
MAX_TIME = 999999
D = int(sys.argv[1])
T = int(sys.argv[2])
functions = [ADD, MUL]
terminals = [Const(T)]

config = TGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=None,
    minimizing_fitness=False,
    ideal_fitness=None,
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

hyperparameters = SimpleTGPHyperparameters(
    lmbda=1,
    k=1,
    strict_selection=False,
    check_size=False,
    max_depth=D,
    multi=True
)

if hyperparameters.multi:
    appendix = "multi"
else:
    appendix = "single"

problem = MaxPlusMul(d=D, t=T)
config.ideal_fitness = problem.ideal
config.global_seed = int(time.time_ns())
tgp = SimpleQdTGP(functions, terminals, config, hyperparameters)
tgp.evolve(problem)

print(f"{D},simple_qd_tgp_{appendix},{tgp.generation_number}")
