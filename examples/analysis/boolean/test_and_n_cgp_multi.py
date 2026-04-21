import sys
from src.analysis.benchmarks.boolean import Conjunction, DatasetType
from src.analysis.models.simple_cgp import SimpleCGP, SimpleCGPConfig, MutationType
from src.gp.tiny_cgp import *
from src.gp.functions import AND, NOTA

MAX_GENERATIONS = 1000000
MAX_TIME = 9999999
N_MIN = 2
N_MAX = 50
K = 1.3
S = 3
T = 30
NEGATED_VARIABLES = False
DATASET_TYPE = DatasetType.SAMPLE


for n in range(N_MIN, N_MAX + 1):

    MAX_ARITY = 2
    NUM_OUTPUTS = 1
    NUM_FUNCTION_NODES = S * n + 1
    NUM_GENES = (MAX_ARITY + 1) * NUM_FUNCTION_NODES + NUM_OUTPUTS
    LEVELS_BACK = NUM_FUNCTION_NODES
    MUTATION_RATE = 1 / NUM_GENES

    if NEGATED_VARIABLES:
        NUM_TERMINALS = 2 * n
    else:
        NUM_TERMINALS = n

    functions = [AND]
    terminals = [Var(i) for i in range(NUM_TERMINALS)]

    config = SimpleCGPConfig(
        num_jobs=1,
        max_generations=MAX_GENERATIONS,
        stopping_criteria=0,
        minimizing_fitness=True,
        ideal_fitness=0,
        silent_algorithm=True,
        silent_evolver=True,
        minimalistic_output=True,
        num_functions=len(functions),
        max_arity=MAX_ARITY,
        num_inputs=NUM_TERMINALS,
        num_outputs=NUM_OUTPUTS,
        report_interval=1,
        max_time=MAX_TIME,
        mutation_type=MutationType.SAM,
        global_seed=None,
        checkpoint_interval=9999999,
        checkpoint_dir='../checkpoint',
        experiment_name='and_cgp'
    )

    hyperparameters = CGPHyperparameters(
        mu=1,
        lmbda=1,
        population_size=2,
        num_function_nodes=NUM_FUNCTION_NODES,
        levels_back=LEVELS_BACK,
        mutation_rate=MUTATION_RATE,
        strict_selection=True
    )

    if config.mutation_type == MutationType.SAM:
        appendix = "sam"
    else:
        appendix = "prob"

    for _ in range(T):
        problem = Conjunction(n=n, negated_vars=NEGATED_VARIABLES, k=K, dataset_type=DATASET_TYPE)
        config.ideal_fitness = problem.ideal
        config.global_seed = int(time.time_ns())
        cgp = SimpleCGP(functions, terminals, config, hyperparameters)
        program = cgp.evolve(problem)
        print(f"{n},simple_cgp_{appendix},{cgp.generation_number},{problem.calc_generalisation_error(program.genome, cgp)}")
