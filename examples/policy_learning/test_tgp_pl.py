"""
Example module to test TGP with policy search problems.
Evolves a policy for the Gymnasium Lunar Lander environment.

TGP is used with multiple trees.

https://gymnasium.farama.org/environments/box2d/lunar_lander/

The Lunar Lander has the following specifications that are adapted to
the GP mode in this example:

Action space: Discrete(4)

Observation space: Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ],
                       [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
"""

from math import sqrt, pi
import gymnasium as gym

from src.gp.tinyverse import Var, Const
from src.gp.tiny_tgp import TGPHyperparameters, TinyTGP, GPConfig
from src.gp.functions import ADD, SUB, MUL, DIV, lAND, lOR, lNAND, lNOR, lNOT, IF, LT, GT
from src.gp.problem import PolicySearch
import warnings
import numpy

if numpy.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

env = gym.make("LunarLander-v3")
wrapped_env = gym.wrappers.FlattenObservation(env)

NUM_INPUTS = wrapped_env.observation_space.shape[0]
functions = [ADD, SUB, MUL, DIV, lAND, lOR, lNAND, lNOR, lNOT, IF, LT, GT]
terminals = [Var(i) for i in range(NUM_INPUTS)] + [
    Const(1),
    Const(2),
    Const(sqrt(2)),
    Const(pi),
    Const(0.5),
]

config = GPConfig(
    num_jobs=1,
    max_generations=50,
    stopping_criteria=300,
    minimizing_fitness=False,
    ideal_fitness=300,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=4,
    report_interval=1,
    max_time=500,
    global_seed=42,
    checkpoint_interval=100,
    checkpoint_dir='examples/checkpoint',
    experiment_name='pl_tgp'
)

hyperparameters = TGPHyperparameters(
    pop_size=50,
    max_size=25,
    max_depth=5,
    cx_rate=0.9,
    mutation_rate=0.3,
    tournament_size=2,
    erc=False
)

problem = PolicySearch(env=env, ideal_=300, minimizing_=False)
tgp = TinyTGP(functions, terminals, config, hyperparameters)
policy = tgp.evolve(problem)

print(tgp.expression(policy.genome))

env = gym.make("LunarLander-v3", render_mode="human")
problem = PolicySearch(env=env, ideal_=100, minimizing_=False)
problem.evaluate(policy.genome, tgp, num_episodes=1, wait_key=True)
