"""
Example module to test GE with policy search problems.
Evolves a policy for the Gymnasium Lunar Lander environment.

https://gymnasium.farama.org/environments/box2d/lunar_lander/

The Lunar Lander has the following specifications that are adapted to
the GP mode in this example:

Action space: Discrete(4)

Observation space: Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ],
                       [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
"""

from math import sqrt, pi

from src.gp.tiny_ge import *
from src.gp.functions import *
from src.gp.problem import PolicySearch
import warnings
import numpy

if numpy.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

env = gym.make("LunarLander-v3")

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
    max_time=5000,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='pl_ge'
)

hyperparameters = GEHyperparameters(
    pop_size=100,
    genome_length=50,
    codon_size=100,
    cx_rate=0.95,
    mutation_rate=0.25,
    tournament_size=2,
    penalty_value=-99999,
)

problem = PolicySearch(env=env, ideal_=300, minimizing_=False)

functions = [ADD, SUB, MUL, DIV, lAND, lOR, lNAND, lNOR, lNOT, IF, LT, GT]
arguments = ["a", "b", "c", "d", "e", "f", "g", "h"]  # Inputs for the functions
grammar = {
    "<expr>": ["[<lfun>, <lfun>, <lfun>, <lfun>]"],
    "<lfun>": ["IF(<lfun>, <fun>, <fun>)", "<logic>(<fun>, <fun>)"],
    "<logic>": ["AND", "OR", "NAND", "NOR", "NOT"],
    "<lfun>": ["LT(<cvar>, <cvar>)", "GT(<cvar>, <cvar>)"],
    "<fun>": ["ADD(<fun>, <fun>)", "SUB(<fun>, <fun>)", "MUL(<fun>, <fun>)", "DIV(<fun>, <fun>)", "<lfun>", "<cvar>"],
    "<cvar>": ["<var>"],# ["<const>", "<var>"],
    "<const>" : ["1", "2", "0.5", str(pi), str(sqrt(2))],
    "<var>" : arguments
}

ge = TinyGE(functions, grammar, arguments, config, hyperparameters)
policy = ge.evolve(problem)
print(ge.expression(policy.genome))

env = gym.make("LunarLander-v3", render_mode="human")
problem = PolicySearch(env=env, ideal_=100, minimizing_=False)
problem.evaluate(policy.genome, ge, num_episodes=1, wait_key=True)
