"""
Example module to test TGP with symbolic regression problems.

Attempts to evolve a solution for the Koza-1 benchmkark which is
a quartic polynomial: x^4 + x^3 + x^2 + x

The problem is specified in the following paper:
https://dl.acm.org/doi/10.1145/2330163.2330273

Please note: This benchmark is nowadays considered a toy problem and
no serious benchmark. It only serves as an example for SR as an application
domain for TinyverseGP:
"""

from src.gp.tiny_tgp import TinyTGP, TGPHyperparameters
from src.gp.functions import ADD, SUB, MUL, DIV
from src.gp.loss import absolute_distance
from src.gp.problem import BlackBox
from src.benchmark.symbolic_regression.sr_benchmark import SRBenchmark
from src.gp.tinyverse import GPConfig, Var, Const
from src.hpo.hpo import SMACInterface


config = GPConfig(
    num_jobs=1,
    max_generations=100,
    stopping_criteria=1e-6,
    minimizing_fitness=True,  # this should be used from the problem instance
    ideal_fitness=1e-6,  # this should be used from the problem instance
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1,
    max_time=60,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='sr_cgp'
)

hyperparameters = TGPHyperparameters(
    pop_size=100,
    max_size=25,
    max_depth=5,
    cx_rate=0.9,
    mutation_rate=0.3,
    tournament_size=2,
    erc=True
)

loss = absolute_distance
benchmark = SRBenchmark()
data, actual = benchmark.generate("KOZA1")
functions = [ADD, SUB, MUL, DIV]
terminals = [Var(0), Const(1)]
trials = 25

problem = BlackBox(data, actual, loss, 1e-6, True)
tgp = TinyTGP(functions, terminals, config, hyperparameters)

interface = SMACInterface()

## Perform HPO via SMAC
opt_hyperparameters = interface.optimise(tgp,trials)
print(opt_hyperparameters)

config.silent_algorithm=False
config.silent_evolver=False
tgp = TinyTGP(functions, terminals, config, opt_hyperparameters)
tgp.evolve(problem)