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

from gymnasium.wrappers import FlattenObservation

from src.gp.tiny_lgp import *
from src.gp.functions import *
from src.gp.problem import PolicySearch
import warnings
import numpy

if numpy.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

env = gym.make("LunarLander-v3")
wrapped_env = FlattenObservation(env)

NUM_INPUTS = wrapped_env.observation_space.shape[0]
functions = [ADD, SUB, MUL, DIV, lAND, lOR, lNAND, lNOR, lNOT] #, IF, LT, GT]
terminals = [Var(i) for i in range(NUM_INPUTS)]#*2 + [ Const(1), Const(2), Const(sqrt(2)), Const(pi), Const(0.5), ]

hyperparameters = LGPHyperparameters(
        mu=500,
        macro_variation_rate=0.75,
        micro_variation_rate=0.25,
        insertion_rate=0.5,
        max_segment=10,
        reproduction_rate=0.5,
        branch_probability=0.05,
        p_register = 0.15,
        max_len = 100,
        initial_max_len = 30,
        erc = False,
        default_value = 0.0,
        protection = 1e10
    )
config = LGPConfig(
        num_jobs=1,
        max_generations=1000, #- hyperparameters.mu,
        stopping_criteria=100,
        minimizing_fitness=False,
        ideal_fitness=100,
        silent_algorithm=False,
        silent_evolver=False,
        minimalistic_output=True,
        report_interval=5,
        max_time=1200,
        num_outputs=4,
        num_registers=5,
        global_seed=None,
        checkpoint_interval=100000,
        checkpoint_dir="checkpoints",
        experiment_name="my_experiment",
)

problem = PolicySearch(env=env, ideal_=300, minimizing_=False)
lgp = TinyLGP(functions, terminals, config, hyperparameters)
policy = lgp.evolve(problem)

#env = gym.make("LunarLander-v3", render_mode="human")
#problem = PolicySearch(env=env, ideal_=100, minimizing_=False)
#problem.evaluate(policy.genome, lgp, num_episodes=1, wait_key=True)
