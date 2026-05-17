"""
Example module to test CGP with policy search problems.
Evolves a policy for Breakout from the MinAtar benchmark:
https://github.com/kenjyoung/MinAtar
"""

from src.benchmark.policy_search.pl_benchmark import PLBenchmark, ALEArgs, MinAtarArgs
from src.gp.tiny_cgp import *
from src.gp.problem import PolicySearch
from src.gp.functions import *
from src.gp.tinyverse import Var
import warnings
import numpy

from minatar import gym as gym_ma
import gymnasium as gym

gym_ma.register_envs()

if numpy.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

NUM_GENERATIONS = 10000
MAX_TIME = 3600
IDEAL = 100
NUM_EPISODES = 10
MAX_EPISODE_STEPS = 2500
MAX_STEPS = 2e8
GAME = 'MinAtar/Breakout-v1'

minatar_args = MinAtarArgs(
    max_steps=MAX_STEPS,
    max_episode_steps=MAX_EPISODE_STEPS,
    difficulty=0,
    flatten_obs=True,
    use_minimal_action_set=False)

env = gym.make(id=GAME, max_episode_steps=MAX_EPISODE_STEPS, render_mode="array")
benchmark = PLBenchmark(env, args_=minatar_args)
wrapped_env = benchmark.wrapped_env
num_inputs = benchmark.len_observation_space()
num_outputs = benchmark.len_action_space()

functions_ext = [ADD, SUB2F, MUL, DIV, INV, ABS, SIN, COS, TAN, ARCSIN, ARCCOS, ARCTAN, LOG, SQR, SQRT,
                 CEIL, FLOOR, lAND, lOR, lNAND, lNOR, lNOT, lXOR, LT, LTE, GT, GTE, EQ, NEQ, MIN, MAX, IF, IFLEZ, IFGTZ]
functions_red = [ADD, SUB2F, MUL, DIV, lAND, lOR, lNAND, lNOR, lNOT, LT, GT, EQ, MIN, MAX, IF]
functions = functions_red
terminals = [Var(i) for i in range(num_inputs)]

config = CGPConfig(
    num_jobs=1,
    max_generations=NUM_GENERATIONS,
    stopping_criteria=IDEAL,
    minimizing_fitness=False,
    ideal_fitness=IDEAL,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=3,
    num_inputs=num_inputs,
    num_outputs=num_outputs,
    report_interval=1000,
    max_time=99999999,
    global_seed=42,
    checkpoint_interval=1,
    checkpoint_dir='checkpoint',
    experiment_name='pl_cgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=8,
    population_size=9,
    levels_back=50,
    num_function_nodes=50,
    mutation_rate=0.02,
    strict_selection=False,
)

problem = PolicySearch(env=wrapped_env, ideal_=config.ideal_fitness, minimizing_=False, num_episodes_=NUM_EPISODES,
                       max_steps_=MAX_STEPS)
cgp = TinyCGP(functions, terminals, config, hyperparameters)
policy = cgp.evolve(problem)
env.close()

env = gym.make(id=GAME, render_mode="human")
problem = PolicySearch(env=env, ideal_=config.ideal_fitness, minimizing_=False)
problem.evaluate(policy.genome, cgp, num_episodes=1, wait_key=True)
env.close()
