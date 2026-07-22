from src.gp.tiny_cgp import *
from src.gp.functions import *
from src.gp.loss import *
from src.gp.tiny_3ge import TreeGEConfig, TreeGEHyperparameters
from src.hpo.hpo_model import HPOModel
from src.gp.functions import ADD, SUB, MUL, DIV, EXP, LOG, SQR, CUBE

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
dataset = ["192_vineyard"]

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
treege_config = TreeGEConfig(
    num_jobs=1,
    max_generations=100,
    stopping_criteria=1e-6,
    minimizing_fitness=True,
    ideal_fitness=1e-6,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1,
    max_time=6000,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='sr_3ge'
)

treege_hyperparams = TreeGEHyperparameters(
    pop_size=32,
    min_depth=2,
    max_depth=6,
    codon_size=256,
    cx_rate=0.9,
    mutation_rate=0.1,
    tournament_size=2,
    penalty_value=99999,
)

functions = [ADD, SUB, MUL, DIV, EXP, LOG, SQR, CUBE]

grammar = {
    "<expr>": [
        "ADD(<expr>, <expr>)",
        "SUB(<expr>, <expr>)",
        "MUL(<expr>, <expr>)",
        "DIV(<expr>, <expr>)",
        "EXP(<expr>)",
        "LOG(<expr>)",
        "SQR(<expr>)",
        "CUBE(<expr>)",
        "<const>",
        "<var>",
    ],
    "<const>": ["1", "0.5", "3.14159", "1.41421", "2", "3", "4", "5", "6", "7", "8", "9"],
    "<var>": [],
}

# ---------------------------------------------------------------------
# Run HPO
# ---------------------------------------------------------------------
hpo_model = HPOModel(
    representation="3GE",
    config=treege_config,
    hyperparameters=treege_hyperparams,
    dataset=dataset,
    functions=functions,
    grammar=grammar,
    loss=mean_squared_error,
)

results = hpo_model.run_smac(train_size=0.75, n_trials=20)

# ---------------------------------------------------------------------
# Summarize results
# ---------------------------------------------------------------------
# print("\n=== HPO Results Summary ===")

# # Extract numeric results if SMAC returns a list of scores
# if isinstance(results, list) or isinstance(results, np.ndarray):
#     mean_val = statistics.mean(results)
#     std_val = statistics.stdev(results) if len(results) > 1 else 0.0
#     min_val = min(results)
#     max_val = max(results)

#     print(f"Number of trials: {len(results)}")
#     print(f"Mean: {mean_val:.4f}")
#     print(f"Std Dev: {std_val:.4f}")
#     print(f"Min: {min_val:.4f}")
#     print(f"Max: {max_val:.4f}")
#     print(f"Median: {statistics.median(results):.4f}")

# else:
    # If SMAC returns a dict with metrics
# print("Raw results object:")
# print(results)

# print("\nAll trial scores:")
# print(results)