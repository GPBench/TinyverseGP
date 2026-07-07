"""
Example script to test CGP with symbolic regression problems and multiple jobs.
"""

from src.gp.tiny_cgp import *
from src.gp.problem import BlackBox
from src.benchmark.symbolic_regression.sr_benchmark import SRBenchmark
from src.gp.functions import *
from src.gp.loss import *
from src.gp.tinyverse import Var, Const

functions = [ADD, SUB, MUL, DIV]
terminals = [Var(0), Const(1)]

config = CGPConfig(
    num_jobs=10,
    max_generations=1000,
    stopping_criteria=1e-6,
    minimizing_fitness=True,
    ideal_fitness=1e-6,
    silent_algorithm=True,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=1,
    num_outputs=1,
    report_interval=1,
    max_time=60,
    global_seed=None,
    checkpoint_interval=10,
    checkpoint_dir='checkpoint',
    experiment_name='sr_cgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=32,
    population_size=33,
    num_function_nodes=10,
    levels_back=len(terminals),
    mutation_rate=0.1,
    strict_selection=True,
)

loss = absolute_distance
benchmark = SRBenchmark()
data, actual = benchmark.generate("KOZA3")

problem = BlackBox(data, actual, loss, 1e-6, True)

cgp = TinyCGP(functions, terminals, config, hyperparameters)
cgp.evolve(problem)
