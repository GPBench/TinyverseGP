"""
Example module to test TinyverseGP on SRBench.

More information about SRBench can be obtained here:

https://cavalab.org/srbench/
https://github.com/cavalab/srbench/tree/master
"""

from src.benchmark.symbolic_regression.srbench import SRBench
import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.gp.tinyverse import GPConfig
from src.gp.tiny_ge import TinyGE, GEHyperparameters
from src.gp.functions import ADD, SUB, MUL, DIV, EXP, LOG, SQRT, SQR, CUBE
from src.gp.tiny_cgp import CGPConfig, CGPHyperparameters
from src.gp.tiny_tgp import TGPHyperparameters, TGPConfig
from src.gp.tiny_lgp import LGPHyperparameters, LGPConfig

MAXTIME = 3600  # 1 hour
MAXGEN = 100
POPSIZE = 100

group_datasets = [
    ["192_vineyard", "1028_SWD"],  # "522_pm10, "678_visualizing_environmental""
    ["1199_BNG_echoMonths", "210_cloud", "1089_USCrime", "1193_BNG_lowbwt"],
    [
        "557_analcatdata_apnea1",
        "650_fri_c0_500_50",
        "579_fri_c0_250_5",
        "606_fri_c2_1000_10",
    ],
]

functions = [ADD, SUB, MUL, DIV, EXP, LOG, SQR, CUBE]
terminals = [1, 0.5, np.pi, np.sqrt(2)]

grammar = {
    "<expr>": [
        "ADD(<expr>, <expr>)",
        "SUB(<expr>, <expr>)",
        "MUL(<expr>, <expr>)",
        "DIV(<expr>, <expr>)",
        "EXP(<expr>)",
        "LOG(<expr>)",
        "SQR(<expr>)",
        # "CUBE(<expr>)",
        "<const>",
        "<var>",
    ],
    "<const>": ["1", "0.5", "3.14159", "1.41421"],  # ~pi, ~sqrt(2)
    "<var>": ["x"],
}

# Set up hyperparameters for TGP and CGP
tgp_hyperparams = TGPHyperparameters(
    max_depth=8,
    max_size=100,
    pop_size=POPSIZE,
    tournament_size=3,
    mutation_rate=0.2,
    cx_rate=0.9,
    erc=False,  # ephemeral random constants
)
cgp_hyperparams = CGPHyperparameters(
    mu=2,
    lmbda=10,
    num_function_nodes=10,
    population_size=POPSIZE,
    levels_back=10,
    strict_selection=True,
    mutation_rate=0.3,
)
lgp_hyperparams = LGPHyperparameters(
    mu=POPSIZE,
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
    penalization_validity_factor=0.0
)

ge_hyperparams = GEHyperparameters(
    pop_size=100,
    genome_length=40,
    codon_size=1000,
    cx_rate=0.9,
    mutation_rate=0.1,
    tournament_size=2,
    penalty_value=99999,
)

#   Set up configurations for TGP and CGP
tgp_config = TGPConfig(
    num_jobs=1,
    max_generations=MAXGEN,
    stopping_criteria=1e-6,
    minimizing_fitness=True,
    ideal_fitness=1e-16,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1000000,
    max_time=MAXTIME,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir="examples/checkpoint",
    experiment_name="srbench_tgp",
)
cgp_config = CGPConfig(
    num_jobs=1,
    max_generations=MAXGEN,
    stopping_criteria=1e-16,
    minimizing_fitness=True,
    ideal_fitness=1e-16,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=1,
    num_outputs=1,
    # num_function_nodes=30,
    report_interval=10,
    max_time=MAXTIME,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir="examples/checkpoint",
    experiment_name="srbench_cgp",
)
lgp_config = LGPConfig(
    num_jobs=1,
    max_generations=MAXGEN,
    stopping_criteria=1e-6,
    minimizing_fitness=True,
    ideal_fitness=1e-6,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    report_interval=100000000000,
    max_time=500,
    num_registers=8,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir="examples/checkpoint",
    experiment_name="srbench_lgp",
)

ge_config = GPConfig(
    num_jobs=1,
    max_generations=100,
    stopping_criteria=1e-6,
    minimizing_fitness=True,  # this should be used from the problem instance
    ideal_fitness=1e-6,  # this should be used from the problem instance
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1,
    max_time=60,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='sr_ge'
)

for g in group_datasets:
    for d in g:
        print(f"Running dataset: {d}\n")
        X, y = fetch_data(d, return_X_y=True)
        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75)

        tgp = SRBench(
            "TGP",
            tgp_config,
            tgp_hyperparams,
            functions=functions,
            terminals=terminals,
            scaling_=False,
        )
        cgp = SRBench(
            "CGP",
            cgp_config,
            cgp_hyperparams,
            functions=functions,
            terminals=terminals,
            scaling_=False,
        )
        lgp = SRBench(
            "LGP",
            lgp_config,
            lgp_hyperparams,
            functions=functions,
            terminals=terminals,
            scaling_=False,
        )

        ge = SRBench(
            "GE",
            ge_config,
            ge_hyperparams,
            functions=functions,
            terminals=terminals,
            scaling_=False,
            # grammar=grammar
        )

        cgp.fit(
            train_X, train_y
        )
        print(cgp.get_model())
        print(f"cgp train score: {cgp.score(train_X, train_y)}")
        print(f"cgp test score: {cgp.score(test_X, test_y)}")
        tgp.fit(train_X, train_y)
        print(tgp.get_model())
        print(f"tgp train score: {tgp.score(train_X, train_y)}")
        print(f"tgp test score: {tgp.score(test_X, test_y)}")
        print("=" * 50)
        lgp.fit(train_X, train_y)
        print(lgp.get_model())
        print(f"lgp train score: {lgp.score(train_X, train_y)}")
        print(f"lgp test score: {lgp.score(test_X, test_y)}")
        print("=" * 50)
        ge.fit(train_X, train_y)
        print(ge.get_model())
        print(f"ge train score: {ge.score(train_X, train_y)}")
        print(f"ge test score: {ge.score(test_X, test_y)}")  #
