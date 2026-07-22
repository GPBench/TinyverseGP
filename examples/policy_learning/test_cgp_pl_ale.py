"""
Example module to test CGP with policy search problems.
Evolves a policy for Pong from the Gymnasium Atari Learning Environment:

https://ale.farama.org/
https://ale.farama.org/environments/

https://ale.farama.org/environments/pong/

Pong has the following specifications that are adapted to
the GP mode in this example:

Action space: Discrete(6)

Observation space: Box(0, 255, (210, 160, 3), uint8)
"""
from src.benchmark.policy_search.pl_benchmark import PLBenchmark, ALEArgs
from src.gp.tiny_cgp import TinyCGP, CGPHyperparameters, CGPConfig
from src.gp.problem import PolicySearch
from src.gp.functions import ADD, SUB2F, MUL, DIV, INV, ABS, SIN, COS, TAN, ARCSIN, ARCCOS, ARCTAN, LOG, SQR, SQRT, CEIL, FLOOR, lAND, lOR, lNAND, lNOR, lNOT, lXOR, LT, LTE, GT, GTE, EQ, NEQ, MIN, MAX, IF, IFLEZ, IFGTZ

import warnings
import numpy
import gymnasium as gym


if numpy.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

gym.envs.registry.keys()

MAX_GENERATIONS = 9999999
IDEAL = 1000
GAME = "ALE/Pong-v5"
NUM_EPISODES = 10
MAX_STEPS = 2e8

ale_args = ALEArgs(
    noop_max=30,
    frame_skip=1,
    screen_size=84,
    grayscale_obs=True,
    terminal_on_life_loss=False,
    scale_obs=False,
    frame_stack=1,
    repeat_action_probability=0.0,
    max_steps=MAX_STEPS,
    full_action_space=False,
    difficulty=0,
    frames_per_step=4,
    max_episode_steps=2500,
    flatten_obs=True
)

benchmark = PLBenchmark(env_=gym.make(GAME), args_=ale_args)
env = benchmark.wrapped_env
functions_ext = [ADD, SUB2F, MUL, DIV, INV, ABS, SIN, COS, TAN, ARCSIN, ARCCOS, ARCTAN, LOG, SQR, SQRT,
                 CEIL, FLOOR, lAND, lOR, lNAND, lNOR, lNOT, lXOR, LT, LTE, GT, GTE, EQ, NEQ, MIN, MAX, IF, IFLEZ, IFGTZ]
functions_red = [ADD, SUB2F, MUL, DIV, lAND, lOR, lNAND, lNOR, lNOT, LT, GT, EQ, MIN, MAX, IF]
functions = functions_red
terminals = benchmark.gen_terminals()
num_inputs = benchmark.len_observation_space()
num_outputs = benchmark.len_action_space()

config = CGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
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
    report_interval=1,
    max_time=9999999,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='checkpoint',
    experiment_name='pl_cgp_ale'
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

problem = PolicySearch(env=env, ideal_=IDEAL, minimizing_=False, num_episodes_=NUM_EPISODES,
                       max_steps_=MAX_STEPS)
cgp = TinyCGP(functions, terminals, config, hyperparameters)
policy = cgp.evolve(problem)
env.close()

env = gym.make(id=GAME, render_mode="human", full_action_space = ale_args.full_action_space)
problem = PolicySearch(env=env, ideal_=IDEAL, minimizing_=False)
problem.evaluate(policy.genome, cgp, num_episodes=1, wait_key=True)
env.close()
