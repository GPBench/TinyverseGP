
from src.benchmark.policy_search.pl_benchmark import PLBenchmark
from src.gp.tiny_cgp import *
from src.gp.problem import PolicySearch
from src.gp.functions import *
from src.gp.tinyverse import Var
import warnings
import numpy

from minatar import gym
gym.register_envs()

if numpy.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

env = gym.BaseEnv(game='breakout', render_mode="array", use_minimal_action_set=True, display_time=50)
benchmark = PLBenchmark(env, ale_=False, ale_args=None, flatten_obs_=True)
wrapped_env = benchmark.wrapped_env
num_inputs = benchmark.len_observation_space()
num_outputs = benchmark.len_action_space()

functions = [ADD, MUL, DIV, AND, OR, NAND, NOR, NOTA, BUFA, XOR, XNOR, IF, LT, GT]
terminals = [Var(i) for i in range(num_inputs)]

config = CGPConfig(
    num_jobs=1,
    max_generations=10000,
    stopping_criteria=100,
    minimizing_fitness=False,
    ideal_fitness=100,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=3,
    num_inputs=num_inputs,
    num_outputs=num_outputs,
    report_interval=1,
    max_time=999999,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='checkpoint',
    experiment_name='pl_cgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=1,
    population_size=2,
    num_function_nodes=100,
    levels_back=99999,
    mutation_rate=0.01,
    strict_selection=False,
)


problem = PolicySearch(env=env, ideal_=100, minimizing_=False, num_episodes_=20)
cgp = TinyCGP(functions, terminals, config, hyperparameters)
policy = cgp.evolve(problem)
env.close()

env = gym.BaseEnv(game='breakout', render_mode="human", use_minimal_action_set=True, display_time=50)
problem = PolicySearch(env=env, ideal_=100, minimizing_=False)
problem.evaluate(policy.genome, cgp, num_episodes=1, wait_key=True)
env.close()
