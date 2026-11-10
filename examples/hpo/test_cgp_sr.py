"""
Example script to perform HPO via SMAC for CGP on a simple (toy) symbolic regression problem.
"""

from src.gp.tiny_cgp import TinyCGP, CGPConfig, CGPHyperparameters
from src.gp.problem import BlackBox
from src.benchmark.symbolic_regression.sr_benchmark import SRBenchmark
from src.gp.functions import ADD, SUB, MUL, DIV
from src.gp.loss import absolute_distance
from src.gp.tinyverse import Var, Const
from src.hpo.hpo import SMACInterface

functions = [ADD, SUB, MUL, DIV]
terminals = [Var(0), Const(1)]

config = CGPConfig(
    num_jobs=1,
    max_generations=10,
    stopping_criteria=1e-6,
    minimizing_fitness=True,
    ideal_fitness=1e-6,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=1,
    num_outputs=1,
    num_function_nodes=10,
    report_interval=1,
    max_time=60,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='sr_cgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=32,
    population_size=33,
    levels_back=len(terminals),
    mutation_rate=0.1,
    strict_selection=True,
)

loss = absolute_distance
benchmark = SRBenchmark()
data, actual = benchmark.generate("KOZA3")
trials = 25

problem = BlackBox(data, actual, loss, 1e-6, True)
cgp = TinyCGP(functions, terminals, config, hyperparameters)
interface = SMACInterface()

## Perform HPO via SMAC
opt_hyperparameters = interface.optimise(cgp, trials)
print(opt_hyperparameters)

config.silent_algorithm=False
config.silent_evolver=False
cgp = TinyCGP(functions, terminals, config, opt_hyperparameters)
cgp.evolve(problem)
