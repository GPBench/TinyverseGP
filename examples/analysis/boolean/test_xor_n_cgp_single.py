import sys

from src.analysis.benchmarks.boolean import ExclusiveDisjunction, DatasetType
from src.analysis.models.simple_cgp import SimpleCGP, SimpleCGPConfig, MutationType
from src.gp.tiny_cgp import *
from src.gp.functions import AND, XOR

MAX_GENERATIONS = 1000000
MAX_TIME = 9999999
N = int(sys.argv[1])
S = 1
K = 1.3
MAX_ARITY = 2
NUM_OUTPUTS = 1
NUM_FUNCTION_NODES = S * N
NUM_GENES = (MAX_ARITY + 1) * NUM_FUNCTION_NODES + NUM_OUTPUTS
MUTATION_RATE = 1 / NUM_GENES
NUM_FUNCTION_NODES = N + 1
LEVELS_BACK = NUM_FUNCTION_NODES
NEGATED_VARIABLES = False
DATASET_TYPE = DatasetType.SPLIT

if NEGATED_VARIABLES:
    NUM_TERMINALS = 2 * N
else:
    NUM_TERMINALS = N

functions = [XOR]
terminals = [Var(i) for i in range(NUM_TERMINALS)]

config = SimpleCGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=None,
    minimizing_fitness=True,
    ideal_fitness=None,
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
    experiment_name='max_tgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=1,
    population_size=2,
    num_function_nodes=N + 1,
    levels_back=N + 1,
    mutation_rate=MUTATION_RATE,
    strict_selection=False
)

if config.mutation_type == MutationType.SAM:
    appendix = "sam"
else:
    appendix = "prob"

problem = ExclusiveDisjunction(n=N, negated_vars=NEGATED_VARIABLES, k=K, dataset_type=DATASET_TYPE)
config.ideal_fitness = problem.ideal
config.global_seed = int(time.time_ns())
cgp = SimpleCGP(functions, terminals, config, hyperparameters)
program = cgp.evolve(problem)

print(f"{N},simple_cgp_{appendix},{cgp.generation_number}, {problem.calc_generalization_error(program.genome, cgp)}")
