import sys
from src.analysis.benchmarks.boolean import Conjunction
from src.analysis.models.simple_tgp import SimpleTGPHyperparameters, SimpleTGP
from src.gp.tiny_cgp import *
from src.gp.functions import AND, NOTA
from src.gp.tiny_tgp import TGPConfig

MAX_GENERATIONS = 1000000
MAX_TIME = 9999999
N = int(sys.argv[1])
K = 1.3
NEGATED_VARIABLES = False
USE_COMPLETE_TRAINING_SET = True

if NEGATED_VARIABLES:
    NUM_TERMINALS = 2 * N
else:
    NUM_TERMINALS = N

functions = [AND]
terminals = [Var(i) for i in range(NUM_TERMINALS)]

config = TGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=0,
    minimizing_fitness=True,
    ideal_fitness=0,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1,
    max_time=MAX_TIME,
    global_seed=None,
    checkpoint_interval=9999999,
    checkpoint_dir='../checkpoint',
    experiment_name='and_cgp'
)

hyperparameters = SimpleTGPHyperparameters(
    lmbda=1,
    k=1,
    strict_selection=False,
    check_size=False,
    max_depth=N * N,
    multi=True
)

if hyperparameters.multi:
    appendix = "multi"
else:
    appendix = "single"

problem = Conjunction(n=N, use_complete_training_set=USE_COMPLETE_TRAINING_SET, negated_vars=NEGATED_VARIABLES, k=K)
config.ideal_fitness = problem.ideal
config.global_seed = int(time.time_ns())
tgp = SimpleTGP(functions, terminals, config, hyperparameters)
program = tgp.evolve(problem)

print(f"{N},simple_tgp_{appendix},{tgp.generation_number}, {problem.calc_generalization_error(program.genome, tgp)}")
