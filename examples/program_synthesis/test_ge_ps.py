"""
Example module to test GE with program synthesis problems.

Attempts to evolve a solution for the "power of two" problem that is provided on
Leetcode.com:

https://leetcode.com/problems/power-of-two/description/

The script wires together a program-synthesis benchmark, the GE configuration and
hyperparameters, a function set and a context-free grammar, and finally runs the
evolutionary loop via :meth:`TinyGE.evolve`.
"""

import warnings


from src.gp.tiny_cgp import GPConfig
from src.gp.problem import ProgramSynthesis
from src.benchmark.program_synthesis.ps_benchmark import PSBenchmark
from src.benchmark.program_synthesis.leetcode.power_of_two import gen_power_of_two
from src.gp.functions import ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, IF, LT, GT
from src.gp.tiny_ge import TinyGE, GEHyperparameters

# Suppress third-party warnings so the run output stays focused on GE progress
warnings.filterwarnings("ignore")

# Run configuration: controls the evolutionary loop, stopping criteria and output
config = GPConfig(
    num_jobs=1,
    max_generations=1000,
    stopping_criteria=100,
    minimizing_fitness=False,
    ideal_fitness=100,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1,
    max_time=60,
)

# GE-specific hyperparameters: genome/codon sizing and variation operator rates
hyperparameters = GEHyperparameters(
    pop_size=100,
    genome_length=40,
    codon_size=1000,
    cx_rate=0.9,
    mutation_rate=0.1,
    tournament_size=2,
    penalty_value=0,
)

# Benchmark generator and the range of inputs used to build the training dataset
generator = gen_power_of_two
n = 10
m = 100

# Build the labelled dataset and wrap it in a program-synthesis problem
benchmark = PSBenchmark(generator, [n, m])
problem = ProgramSynthesis(benchmark.dataset)

# Function set (non-terminals referenced by the grammar) and program arguments
functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, IF, LT, GT]
arguments = ["x"]
# Context-free grammar: <expr> expands recursively into functions, digits or the input x
grammar = {
    "<expr>": [
        "ADD(<expr>, <expr>)",
        "SUB(<expr>, <expr>)",
        "MUL(<expr>, <expr>)",
        "DIV(<expr>, <expr>)",
        "AND(<expr>, <expr>)",
        "OR(<expr>, <expr>)",
        "NAND(<expr>, <expr>)",
        "NOR(<expr>, <expr>)",
        "NOT(<expr>)",
        "IF(<expr>, <expr>, <expr>)",
        "LT(<expr>, <expr>)",
        "GT(<expr>, <expr>)",
        "<d>",
        "<d>.<d><d>",
        "x",
    ],
    "<d>": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
}

# Instantiate the GE model and run the evolutionary search
ge = TinyGE(problem, functions, grammar, arguments, config, hyperparameters)
ge.evolve()
