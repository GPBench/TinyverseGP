from src.benchmark.policy_search.pl_benchmark import PLBenchmark, ALEArgs, PLArgs, MinAtarArgs
from src.gp.tiny_cgp import *
from src.gp.problem import PolicySearch
from src.gp.functions import *
from src.gp.tiny_tgp import TGPHyperparameters, TinyTGP
from src.gp.tinyverse import Var
import warnings
import numpy

from minatar import gym as gym_ma
import gymnasium as gym

gym_ma.register_envs()

if numpy.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

MAX_GENERATIONS = 1000
MAX_TIME = 3600
IDEAL = 100
NUM_EPISODES = 10
NUM_EVAL_EPISODES = 4
MAX_EPISODE_STEPS = 2500
MAX_STEPS = 2e8
GAME = 'MinAtar/Breakout-v1'

minatar_args = MinAtarArgs(max_steps= MAX_STEPS,
                           max_episode_steps= MAX_EPISODE_STEPS,
                           difficulty= 0,
                           flatten_obs= True,
                           use_minimal_action_set = True
                          )

env = gym.make(id=GAME, max_episode_steps = MAX_EPISODE_STEPS, render_mode = "rgb_array")
benchmark = PLBenchmark(env, args_=minatar_args)
wrapped_env = benchmark.wrapped_env
num_inputs = benchmark.len_observation_space()
num_outputs = benchmark.len_action_space()

functions_ext = [ADD, SUB, MUL, DIV, INV, ABS, SIN, COS, TAN, ARCSIN, ARCCOS, ARCTAN, LOG, SQR, SQRT,
                 CEIL, FLOOR, lAND, lOR, lNAND, lNOR, lNOT, LT, LTE, GT, GTE, EQ, NEQ, MIN, MAX, IF, IFLEZ, IFGTZ]
functions_red = [ADD, SUB, MUL, DIV, lAND, lOR, lNAND, lNOR, lNOT, LT, GT, EQ, MIN, MAX, IF]
functions = functions_ext
terminals = [Var(i) for i in range(num_inputs)]

config = GPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=IDEAL,
    minimizing_fitness=False,
    ideal_fitness=IDEAL,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=num_outputs,
    report_interval=10,
    max_time=MAX_TIME,
    global_seed=42,
    checkpoint_interval=100,
    checkpoint_dir='examples/checkpoint',
    experiment_name='pl_tgp'
)

hyperparameters = TGPHyperparameters(
    pop_size=50,
    max_size=50,
    max_depth=6,
    cx_rate=0.9,
    mutation_rate=0.3,
    tournament_size=2,
    erc=False
)


problem = PolicySearch(env=wrapped_env, ideal_=config.ideal_fitness, minimizing_=False, num_episodes_=NUM_EPISODES,
                       max_steps_=MAX_STEPS)
tgp = TinyTGP(functions, terminals, config, hyperparameters)
policy = tgp.evolve(problem)
env.close()

env = gym.make(id=GAME, render_mode="human")
problem = PolicySearch(env=env, ideal_=config.ideal_fitness, minimizing_=False)
problem.evaluate(policy.genome, tgp, num_episodes=1, wait_key=True)
env.close()


