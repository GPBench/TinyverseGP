"""
Example module to test CGP with program synthesis problems.

Attempts to evolve a solution for the numbers of two problem that
is provided on Leetcode.com:

https://leetcode.com/problems/power-of-two/description/

"""

from src.gp.tiny_cgp import TinyCGP, CGPHyperparameters, CGPConfig
from src.gp.problem import ProgramSynthesis
from src.benchmark.program_synthesis.ps_benchmark import PSBenchmark
from src.benchmark.program_synthesis.leetcode.power_of_two import gen_power_of_two
from src.gp.functions import ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, IF, LT, GT
from src.gp.tinyverse import Var, Const

NUM_INPUTS = 1
functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, IF, LT, GT]
terminals = [Var(i) for i in range(NUM_INPUTS)] + [Const(1), Const(2)]

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
    num_inputs=NUM_INPUTS,
    num_outputs=1,
    global_seed=42,
    report_interval=1,
    max_time=60,
    checkpoint_interval=10,
    checkpoint_dir='checkpoint',
    experiment_name='ps_cgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=32,
    population_size=33,
    num_function_nodes=100,
    levels_back=len(terminals),
    mutation_rate=0.1,
    strict_selection=True,
)

generator = gen_power_of_two
n = 10
m = 100

benchmark = PSBenchmark(generator, [n, m])
problem = ProgramSynthesis(benchmark.dataset)
cgp = TinyCGP(functions, terminals, config, hyperparameters)
cgp.evolve(problem)
