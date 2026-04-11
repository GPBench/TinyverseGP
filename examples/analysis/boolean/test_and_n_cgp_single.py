import sys
from src.analysis.benchmarks.boolean import Conjunction
from src.analysis.models.simple_cgp import SimpleCGP, SimpleCGPConfig, MutationType
from src.gp.tiny_cgp import *
from src.gp.functions import AND, NOTA

MAX_GENERATIONS = 1000000
MAX_TIME = 9999999
N = int(sys.argv[1])
K = 1.3
S = 1
MAX_ARITY = 2
NUM_OUTPUTS = 1
NUM_FUNCTION_NODES = S * N
NUM_GENES = (MAX_ARITY + 1) * NUM_FUNCTION_NODES + NUM_OUTPUTS
LEVELS_BACK = NUM_FUNCTION_NODES
MUTATION_RATE = 1 / NUM_GENES
NEGATED_VARIABLES = False
USE_COMPLETE_TRAINING_SET = True

if NEGATED_VARIABLES:
    NUM_TERMINALS = 2 * N
else:
    NUM_TERMINALS = N

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
    strict_selection=False
)

if config.mutation_type == MutationType.SAM:
    appendix = "sam"
else:
    appendix = "prob"

problem = Conjunction(n=N, use_complete_training_set=USE_COMPLETE_TRAINING_SET, negated_vars=NEGATED_VARIABLES, k=K)
config.ideal_fitness = problem.ideal
config.global_seed = int(time.time_ns())
cgp = SimpleCGP(functions, terminals, config, hyperparameters)
program = cgp.evolve(problem)

print(f"{N},simple_cgp_{appendix},{cgp.generation_number},{problem.calc_generalization_error(program.genome, cgp)}")
