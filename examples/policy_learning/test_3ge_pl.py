"""
Example module to test Tree GE with policy search problems.
Evolves a policy for the Gymnasium Lunar Lander environment.

https://gymnasium.farama.org/environments/box2d/lunar_lander/

The Lunar Lander has the following specifications that are adapted to
the GP mode in this example:

Action space: Discrete(4)

Observation space: Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ],
                       [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
"""


from src.gp.tiny_3ge import *
from src.gp.functions import *
from src.gp.problem import PolicySearch
import warnings
import gymnasium as gym
import numpy

if numpy.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

env = gym.make("LunarLander-v3")

config = GPConfig(
    num_jobs=1,
    max_generations=10,
    stopping_criteria=300,
    minimizing_fitness=False,
    ideal_fitness=300,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=4,
    report_interval=1,
    max_time=60,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='pl_ge'
)

hyperparameters = TreeGEHyperparameters(
    pop_size=100,
    min_depth=4,
    max_depth=6,
    codon_size=1000,
    cx_rate=0.9,
    mutation_rate=0.1,
    tournament_size=2,
    penalty_value=99999,
)

problem = PolicySearch(env=env, ideal_=300, minimizing_=False)

functions = [ADD, SUB, MUL, DIV, lAND, lOR, lNAND, lNOR, lNOT, IF, LT, GT]
arguments = ["a", "b", "c", "d", "e", "f", "g", "h"]  # Inputs for the functions
grammar = {
    "<expr>": [
        "ADD(<expr>, <expr>)",
        "SUB(<expr>, <expr>)",
        "MUL(<expr>, <expr>)",
        "DIV(<expr>, <expr>)",
        "AND(<expr>, <expr>)",
        "OR(<expr>, <expr>)",
        "NAND(<expr>, <expr>)",
        "NOR(<expr>, <expr>)",
        "NOT(<expr>)",
        "IF(<expr>, <expr>, <expr>)",
        "LT(<expr>, <expr>)",
        "GT(<expr>, <expr>)",
        "<d>",
        "<d>.<d><d>",
        "1.414",
        "3.141",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
    ],
    "<d>": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
}

treege = Tiny3GE(functions, grammar, arguments, config, hyperparameters)
#treege.print_population(treege.population)
policy = treege.evolve(problem)
# env.close()
env = gym.make("LunarLander-v3", render_mode="human")
problem = PolicySearch(env=env, ideal_=100, minimizing_=False)
problem.evaluate(policy.genome, treege, num_episodes=1, wait_key=True)
# env.close