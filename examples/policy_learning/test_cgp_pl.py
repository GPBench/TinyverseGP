"""
Example module to test CGP with policy search problems.
Evolves a policy for the Gymnasium Lunar Lander environment.

https://gymnasium.farama.org/environments/box2d/lunar_lander/

The Lunar Lander has the following specifications that are adapted to
the GP mode in this example:

Action space: Discrete(4)

Observation space: Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ],
                       [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
"""

from src.gp.tiny_cgp import TinyCGP, CGPHyperparameters, CGPConfig
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from src.gp.problem import PolicySearch
from src.gp.functions import ADD, SUB, MUL, DIV, lAND, lOR, lNAND, lNOR, lNOT, IF, LT, GT
from src.gp.tinyverse import Var, Const
from math import sqrt, pi
import warnings
import numpy

if numpy.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

env = gym.make("LunarLander-v3")
wrapped_env = FlattenObservation(env)
functions = [ADD, SUB, MUL, DIV, lAND, lOR, lNAND, lNOR, lNOT, IF, LT, GT]
terminals = [Var(i) for i in range(wrapped_env.observation_space.shape[0])] + [
    Const(1),
    Const(2),
    Const(sqrt(2)),
    Const(pi),
    Const(0.5),
]

config = CGPConfig(
    num_jobs=1,
    max_generations=50,
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
    report_interval=1,
    max_time=500,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='checkpoint',
    experiment_name='pl_cgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=32,
    population_size=50,
    num_function_nodes=50,
    levels_back=10,
    mutation_rate=0.05,
    strict_selection=True,
)

problem = PolicySearch(env=env, ideal_=100, minimizing_=False)
cgp = TinyCGP(functions, terminals, config, hyperparameters)
policy = cgp.evolve(problem)

env.close()

env = gym.make("LunarLander-v3", render_mode="human")
problem = PolicySearch(env=env, ideal_=100, minimizing_=False)
problem.evaluate(policy.genome, cgp, num_episodes=1, wait_key=True)
env.close()
