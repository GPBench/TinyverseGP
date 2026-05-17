"""
Example script to test Atari5 which is part of PLBench, with TGP, CGP, LGP and GE.
https://arxiv.org/abs/2210.02019
"""

import warnings
import numpy

from src.benchmark.policy_search.pl_benchmark import ALEArgs
from src.benchmark.policy_search.plbench.plbench import PLBench, PLRegressor
from src.gp.functions import *
from src.gp.tiny_cgp import CGPHyperparameters, CGPConfig
from src.gp.tiny_ge import GEHyperparameters
from src.gp.tiny_lgp import LGPHyperparameters, LGPConfig
from src.gp.tiny_tgp import TGPHyperparameters, TGPConfig
from src.gp.tinyverse import Var, GPConfig

if numpy.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

MAX_TIME = 3600  # 1 hour
MAX_GENERATIONS = 100
IDEAL = 1000
NUM_EPISODES = 100
MAX_STEPS = 2e8
POP_SIZE = 50
LAMBDA = 4

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

functions_ext = [ADD, SUB2F, MUL, DIV, INV, ABS, SIN, COS, TAN, ARCSIN, ARCCOS, ARCTAN, LOG, SQR, SQRT,
                 CEIL, FLOOR, lAND, lOR, lNAND, lNOR, lNOT, lXOR, LT, LTE, GT, GTE, EQ, NEQ, MIN, MAX, IF, IFLEZ, IFGTZ]
functions_red = [ADD, SUB2F, MUL, DIV, lAND, lOR, lNAND, lNOR, lNOT, LT, GT, EQ, MIN, MAX, IF]
functions = functions_red

tgp_hyperparams = TGPHyperparameters(
    max_depth=4,
    max_size=30,
    pop_size=POP_SIZE,
    tournament_size=4,
    mutation_rate=0.02,
    cx_rate=0.9,
    erc=None
)

cgp_hyperparams = CGPHyperparameters(
    mu=1,
    lmbda=LAMBDA,
    num_function_nodes=50,
    strict_selection=False,
    mutation_rate=0.02,
    levels_back=50,
    population_size=1+LAMBDA
)

lgp_hyperparams = LGPHyperparameters(
    mu=POP_SIZE,
    macro_variation_rate=0.75,
    micro_variation_rate=0.25,
    insertion_rate=0.5,
    max_segment=15,
    reproduction_rate=0.5,
    branch_probability=0.0,
    p_register=0.5,
    max_len=200,
    initial_max_len=35,
    erc=False,
    default_value=0.0,
    protection=1e10,
    penalization_validity_factor=0.0,
    register_slack=4
)

ge_hyperparams = GEHyperparameters(
    pop_size=POP_SIZE,
    genome_length=100,
    codon_size=1000,
    cx_rate=0.9,
    mutation_rate=0.1,
    tournament_size=2,
    penalty_value=-99999,
)

tgp_config = TGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=100,
    minimizing_fitness=False,
    ideal_fitness=100,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1,
    max_time=MAX_TIME,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir="examples/checkpoint",
    experiment_name="plbench_tgp",
)

cgp_config = CGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=100,
    minimizing_fitness=False,
    ideal_fitness=100,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=3,
    num_inputs=1,
    num_outputs=1,
    report_interval=1,
    max_time=MAX_TIME,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir="examples/checkpoint",
    experiment_name="plbench_cgp",
)

lgp_config = LGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=100,
    minimizing_fitness=False,
    ideal_fitness=100,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    report_interval=1,
    max_time=500,
    num_registers=8,
    global_seed=42,
    checkpoint_interval=1,
    checkpoint_dir="examples/checkpoint",
    experiment_name="srbench_lgp",
)

ge_config = GPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=100,
    minimizing_fitness=False,
    ideal_fitness=100,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1,
    max_time=60,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='sr_ge'
)

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

atari_five = PLBench.AtariFive(args=ale_args)
problems = atari_five.problems

for name, p in problems.items():

    print(f"Running benchmark: {name}")

    num_inputs = p.len_observation_space()
    num_outputs = p.len_action_space()

    tgp_config.num_inputs = num_inputs
    tgp_config.num_outputs = num_outputs

    cgp_config.num_inputs = num_inputs
    cgp_config.num_outputs = num_outputs

    lgp_config.num_inputs = num_inputs
    lgp_config.num_registers = num_outputs
    lgp_config.num_outputs = num_outputs

    ge_config.num_outputs = num_outputs

    terminals = [Var(i) for i in range(num_inputs)]

    tgp = PLRegressor(
        representation_="TGP",
        config_=tgp_config,
        hyperparameters_=tgp_hyperparams,
        functions_=functions,
        terminals_=terminals
    )

    cgp = PLRegressor(
        representation_="CGP",
        config_=cgp_config,
        hyperparameters_=cgp_hyperparams,
        functions_=functions,
        terminals_=terminals
    )

    lgp = PLRegressor(
        representation_="LGP",
        config_=lgp_config,
        hyperparameters_=lgp_hyperparams,
        functions_=functions,
        terminals_=terminals,
        num_episodes_=10
    )

    ge = PLRegressor(
        representation_="GE",
        config_=ge_config,
        hyperparameters_=ge_hyperparams,
        functions_=functions,
        terminals_=terminals,
        num_episodes_=10
    )

    tgp.fit(env=p.env)
    print(f"Reward TGP: {tgp.evaluate()}")

    cgp.fit(env=p.env)
    print(f"Reward CGP: {cgp.evaluate()}")

    lgp.fit(env=p.env)
    print(f"Reward LGP: {lgp.evaluate()}")

    ge.fit(env=p.env)
    if ge.is_valid():
        print(f"Reward GE: {ge.evaluate()}")
    else:
        print(f"GE evaluation cannot be done due to invalid genome")

    p.env.close()
