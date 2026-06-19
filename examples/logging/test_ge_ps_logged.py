"""
Example module to test TGP with program synthesis problems.

Attempts to evolve a solution for the numbers of two problem that
is provided on Leetcode.com:

https://leetcode.com/problems/power-of-two/description/

"""

from src.gp.tiny_cgp import *
from src.gp.problem import ProgramSynthesis
from src.benchmark.program_synthesis.ps_benchmark import PSBenchmark
from src.benchmark.program_synthesis.leetcode.power_of_two import *
from src.gp.functions import *
from src.gp.tiny_tgp import TinyTGP, TGPHyperparameters
from src.gp.tinyverse import Var, Const
import ioh

NUM_INPUTS = 1
functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, IF, LT, GT]
terminals = [Var(i) for i in range(NUM_INPUTS)] + [Const(1), Const(2)]

config = GPConfig(
    global_seed=42,
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
    checkpoint_interval=10,
    checkpoint_dir='checkpoint',
)

hyperparameters = TGPHyperparameters(
    pop_size=100,
    max_size=25,
    max_depth=5,
    cx_rate=0.9,
    mutation_rate=0.3,
    tournament_size=2,
    erc=False  # NOTE: What does this do?
)

generator = gen_power_of_two
n = 10
m = 100

logger = ioh.logger.Analyzer([ioh.logger.trigger.ALWAYS], root='.', folder_name='test_logging',
                             algorithm_name='TGP')

benchmark = PSBenchmark(generator, [n, m])
problem = ProgramSynthesis(benchmark.dataset, logger=logger, fid=1, name="PowerOfTwo")
tgp = TinyTGP(functions, terminals, config, hyperparameters)
tgp.evolve(problem)
