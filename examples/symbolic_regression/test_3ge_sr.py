"""
Example module to test Tree GE with symbolic regression problems.

Attempts to evolve a solution for the Koza-1 benchmkark which is
a quartic polynomial: x^4 + x^3 + x^2 + x

The problem is specified in the following paper:
https://dl.acm.org/doi/10.1145/2330163.2330273

Please note: This benchmark is nowadays considered a toy problem and
no serious benchmark. It only serves as an example for SR as an application
domain for TinyverseGP:
"""

from src.gp.tiny_3ge import Tiny3GE, TreeGEHyperparameters
from src.gp.functions import ADD, SUB, MUL, DIV
from src.gp.loss import absolute_distance
from src.gp.problem import BlackBox
from src.benchmark.symbolic_regression.sr_benchmark import SRBenchmark
from src.gp.tinyverse import GPConfig

config = GPConfig(
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
    max_time=200,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='sr_3ge'
)

hyperparameters = TreeGEHyperparameters(
    pop_size=16,
    min_depth=4,
    max_depth=6,
    codon_size=256,
    cx_rate=0.9,
    mutation_rate=0.1,
    tournament_size=2,
    penalty_value=99999,
)

loss = absolute_distance
benchmark = SRBenchmark()
data, actual = benchmark.generate("KOZA1")
functions = [ADD, SUB, MUL, DIV]
arguments = ["x"]
# grammar = {
#     '<expr>': ['<expr> + <expr>', '<expr> - <expr>', '<expr> * <expr>', '(<expr>)', '<d>', '<d>.<d><d>', 'x'],    # Also possible
#     '<d>': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
# }
grammar = {
    "<expr>": [
        "ADD(<expr>, <expr>)",
        "SUB(<expr>, <expr>)",
        "MUL(<expr>, <expr>)",
        "DIV(<expr>, <expr>)",
        "<d>",
        "<d>.<d><d>",
        "x",
    ],
    "<d>": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
}


problem = BlackBox(data, actual, loss, 1e-6, True)

tree_ge = Tiny3GE(functions, grammar, arguments, config, hyperparameters)

tree_ge.evolve(problem)

#tree_ge.print_individual_tree(tree_ge.best_individual.deriv_tree)
#tree_ge.print_individual(tree_ge.best_individual)