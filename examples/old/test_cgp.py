"""
Example module to test CGP with symbolic regression and policy search problems.
"""

from src.gp.tiny_cgp import *
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from src.gp.problem import BlackBox, PolicySearch
from src.benchmark.symbolic_regression.sr_benchmark import SRBenchmark
from src.gp.functions import *
from src.gp.loss import *
from src.gp.tinyverse import Var, Const
from math import sqrt, pi

print("Koza1 SR Benchmark")
input("Press Enter to begin...")

functions = [ADD, SUB, MUL, DIV]

config = CGPConfig(
    num_jobs=1,
    max_generations=100,
    stopping_criteria=1e-6,
    minimizing_fitness=True,
    ideal_fitness=1e-6,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=1,
    num_outputs=1,
    num_function_nodes=10,
    report_interval=1,
    max_time=60,
)

hyperparameters = CGPHyperparameters(
    mu=100,
    lmbda=100,
    population_size=100,
    levels_back=10,
    mutation_rate=0.1,
    strict_selection=True,
)

config.init()
random.seed(3)

loss = euclidean_distance
benchmark = SRBenchmark()
data, actual = benchmark.generate("KOZA1")
functions = [ADD, SUB, MUL, DIV]
terminals = [Var(0), Const(1)]

problem = BlackBox(data, actual, loss, 1e-6, True)

cgp = TinyCGP(problem, functions, terminals, config, hyperparameters)
cgp.evolve()

print("LunarLander-v3 Benchmark")
input("Press Enter to begin...")

env = gym.make("LunarLander-v3")
wrapped_env = FlattenObservation(env)
functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, IF, LT, GT]
terminals = [Var(i) for i in range(wrapped_env.observation_space.shape[0])] + [
    Const(1),
    Const(2),
    Const(sqrt(2)),
    Const(pi),
    Const(0.5),
]

config = CGPConfig(
    num_jobs=1,
    max_generations=10,
    stopping_criteria=100,
    minimizing_fitness=False,
    ideal_fitness=100,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=3,
    num_inputs=wrapped_env.observation_space.shape[0],
    num_outputs=4,
    num_function_nodes=10,
    report_interval=1,
    max_time=60,
)
config.init()

hyperparameters = CGPHyperparameters(
    mu=10,
    lmbda=10,
    population_size=10,
    levels_back=10,
    mutation_rate=0.1,
    strict_selection=True,
)

problem = PolicySearch(env=env, ideal_=100, minimizing_=False)
cgp = TinyCGP(problem, functions, terminals, config, hyperparameters)
policy = cgp.evolve()
env = gym.make("LunarLander-v3", render_mode="human")
problem = PolicySearch(env=env, ideal_=100, minimizing_=False)
problem.evaluate(policy, cgp, num_episodes=1, wait_key=True)
