import sys
from src.analysis.benchmarks.boolean import Conjunction, DatasetType
from src.analysis.models.simple_tgp import SimpleTGPHyperparameters, SimpleTGP
from src.gp.tiny_cgp import *
from src.gp.functions import AND, NOTA
from src.gp.tiny_tgp import TGPConfig

MAX_GENERATIONS = 1000000
MAX_TIME = 9999999
N_MIN = 2
N_MAX = 50
K = 1.3
T = 30
NEGATED_VARIABLES = False
DATASET_TYPE = DatasetType.SAMPLE

for n in range(N_MIN, N_MAX + 1):

    if NEGATED_VARIABLES:
        NUM_TERMINALS = 2 * n
    else:
        NUM_TERMINALS = n

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
        strict_selection=True,
        check_size=False,
        max_depth=n * n,
        multi=True
    )

    if hyperparameters.multi:
        appendix = "multi"
    else:
        appendix = "single"

    for _ in range(T):

        problem = Conjunction(n=n, negated_vars=NEGATED_VARIABLES, k=K, dataset_type=DATASET_TYPE)
        config.ideal_fitness = problem.ideal
        config.global_seed = int(time.time_ns())
        tgp = SimpleTGP(functions, terminals, config, hyperparameters)
        program = tgp.evolve(problem)
        print(f"{n},simple_tgp_{appendix},{tgp.generation_number}, {problem.calc_generalisation_error(program.genome, tgp)}")
