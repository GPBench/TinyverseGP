"""
Example script to test MinAtar which is part of PLBench, with TGP, CGP, LGP and GE.
https://arxiv.org/abs/1903.03176
https://github.com/kenjyoung/MinAtar
"""

import warnings
import numpy
from src.benchmark.policy_search.pl_benchmark import MinAtarArgs
from src.benchmark.policy_search.plbench.plbench import PLBench, PLRegressor
from src.gp.functions import *
from src.gp.tiny_cgp import CGPHyperparameters, CGPConfig
from src.gp.tiny_ge import GEHyperparameters
from src.gp.tiny_lgp import LGPHyperparameters, LGPConfig
from src.gp.tiny_tgp import TGPHyperparameters, TGPConfig
from src.gp.tinyverse import Var, GPConfig

if numpy.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

functions_ext = [ADD, SUB2F, MUL, DIV, INV, ABS, SIN, COS, TAN, ARCSIN, ARCCOS, ARCTAN, LOG, SQR, SQRT,
                 CEIL, FLOOR, lAND, lOR, lNAND, lNOR, lNOT, lXOR, LT, LTE, GT, GTE, EQ, NEQ, MIN, MAX, IF, IFLEZ, IFGTZ]
functions_red = [ADD, SUB2F, MUL, DIV, lAND, lOR, lNAND, lNOR, lNOT, LT, GT, EQ, MIN, MAX, IF]
functions = functions_red

NUM_GENERATIONS = 10
MAX_TIME = 3600
IDEAL = 100
NUM_EPISODES = 10
MAX_EPISODE_STEPS = 2500
MAX_STEPS = 2e8
POP_SIZE = 50


minatar_args = MinAtarArgs(
    max_steps=MAX_STEPS,
    max_episode_steps=MAX_EPISODE_STEPS,
    difficulty=0,
    flatten_obs=True,
    use_minimal_action_set=True)

minatar = PLBench.MinAtar(args=minatar_args)

tgp_hyperparams = TGPHyperparameters(
    max_depth=4,
    max_size=50,
    pop_size=POP_SIZE,
    tournament_size=4,
    mutation_rate=0.2,
    cx_rate=0.9,
    erc=None
)

cgp_hyperparams = CGPHyperparameters(
    mu=1,
    lmbda=4,
    num_function_nodes=10,
    strict_selection=True,
    mutation_rate=0.2,
    levels_back=10,
    population_size=5
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
    max_generations=NUM_GENERATIONS ,
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
    max_generations=NUM_GENERATIONS ,
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
    max_generations=NUM_GENERATIONS,
    stopping_criteria=100,
    minimizing_fitness=False,
    ideal_fitness=100,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    report_interval=1,
    max_time=MAX_TIME,
    num_registers=8,
    global_seed=42,
    checkpoint_interval=1,
    checkpoint_dir="examples/checkpoint",
    experiment_name="srbench_lgp",
)

ge_config = GPConfig(
    num_jobs=1,
    max_generations=NUM_GENERATIONS,
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
    checkpoint_dir='examples/checkpoint',
    experiment_name='sr_ge'
)

for name, p in minatar.problems.items():

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
        terminals_=terminals,
        num_episodes_=NUM_EPISODES
    )

    cgp = PLRegressor(
        representation_="CGP",
        config_=cgp_config,
        hyperparameters_=cgp_hyperparams,
        functions_=functions,
        terminals_=terminals,
        num_episodes_=NUM_EPISODES
    )

    lgp = PLRegressor(
        representation_="LGP",
        config_=lgp_config,
        hyperparameters_=lgp_hyperparams,
        functions_=functions,
        terminals_=terminals,
        num_episodes_=NUM_EPISODES
    )

    ge = PLRegressor(
        representation_="GE",
        config_=ge_config,
        hyperparameters_=ge_hyperparams,
        functions_=functions,
        terminals_=terminals,
        num_episodes_=NUM_EPISODES
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
