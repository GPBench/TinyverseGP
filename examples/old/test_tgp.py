"""
Example module to test TGP with symbolic regression and policy search problems.
"""

from math import sqrt, pi
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from src.gp.tiny_tgp import *
from src.gp.functions import *
from src.gp.loss import *
from src.gp.problem import BlackBox, PolicySearch
from src.benchmark.symbolic_regression.sr_benchmark import SRBenchmark

print("Koza1 SR Benchmark")
input("Press Enter to begin...")

config = GPConfig(
    num_jobs=1,
    max_generations=100,
    stopping_criteria=1e-6,
    minimizing_fitness=True,  # this should be used from the problem instance
    ideal_fitness=1e-6,  # this should be used from the problem instance
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1,
    max_time=60,
)

hyperparameters = GPHyperparameters(
    pop_size=100,
    max_size=25,
    max_depth=5,
    cx_rate=0.9,
    mutation_rate=0.3,
    tournament_size=2,
)

# random.seed(42)

loss = euclidean_distance
benchmark = SRBenchmark()
data, actual = benchmark.generate("KOZA1")
functions = [ADD, SUB, MUL, DIV]
terminals = [
    Var(0),
    Const(1),
]  # , Const(1), Const(2), Const(sqrt(2)), Const(pi), Const(0.5)]

problem = BlackBox(data, actual, loss, 1e-6, True)

tgp = TinyTGP(problem, functions, terminals, config, hyperparameters)
tgp.evolve()

print("LunarLander-v3 Benchmark")
input("Press Enter to begin...")

env = gym.make("LunarLander-v3")
wrapped_env = FlattenObservation(env)

NUM_INPUTS = wrapped_env.observation_space.shape[0]
functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, IF, LT, GT]
terminals = [Var(i) for i in range(NUM_INPUTS)] + [
    Const(1),
    Const(2),
    Const(sqrt(2)),
    Const(pi),
    Const(0.5),
]

config = GPConfig(
    num_jobs=1,
    max_generations=10,
    stopping_criteria=0.01,
    minimizing_fitness=False,
    ideal_fitness=0.01,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=4,
    report_interval=1,
    max_time=60,
)

hyperparameters = GPHyperparameters(
    pop_size=10,
    max_size=25,
    max_depth=5,
    cx_rate=0.9,
    mutation_rate=0.3,
    tournament_size=2,
)

problem = PolicySearch(env=env, ideal_=100, minimizing_=False)
tgp = TinyTGP(problem, functions, terminals, config, hyperparameters)
policy = tgp.evolve()

env = gym.make("LunarLander-v3", render_mode="human")
problem = PolicySearch(env=env, ideal_=100, minimizing_=False)
problem.evaluate(policy, tgp, num_episodes=1, wait_key=True)
