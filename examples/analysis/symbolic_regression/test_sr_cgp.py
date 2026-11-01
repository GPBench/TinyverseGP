import sys
import time

from analysis.models.simple_cgp import SimpleCGPConfig, SimpleCGPHyperparameters, MutationType, SimpleCGP
from gp.loss import mean_squared_error
from src.analysis.benchmarks.symbolic_regression import TinySR, DatasetType
from src.gp.functions import ADD, MUL, SUB
from src.gp.problem import BlackBox
from src.gp.tinyverse import Var, Const

MAX_GENERATIONS = 10000000
MAX_TIME = 999999
D = int(sys.argv[1])
functions = [ADD, MUL, SUB]
terminals = [Var(0), Const(1), Const(2)]
X_MIN = -1.0
X_MAX = 1.0
C_MIN = 1
C_MAX = D
N = 20
COEFF = True
STEP = 0.02
IDEAL = 1e-2

MAX_ARITY = 2
NUM_GENES = (MAX_ARITY + 1) * D**2 + 1
MUTATION_RATE = 1 / NUM_GENES

config = SimpleCGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=IDEAL,
    minimizing_fitness=True,
    ideal_fitness=IDEAL,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=1,
    num_outputs=1,
    report_interval=1000,
    max_time=MAX_TIME,
    mutation_type=MutationType.PROB,
    global_seed=None,
    checkpoint_interval=9999999,
    checkpoint_dir='../checkpoint',
    experiment_name='max_tgp'
)

hyperparameters = SimpleCGPHyperparameters(
    mu=1,
    lmbda=1,
    population_size=2,
    num_function_nodes=D**2,
    max_active_nodes=D**2,
    levels_back=D**2,
    discard_invalid=True,
    mutation_rate=MUTATION_RATE,
    strict_selection=False,
)

if config.mutation_type == MutationType.SAM:
    appendix = "sam"
else:
    appendix = "prob"

tiny_sr = TinySR(x_min=X_MIN,
                 x_max=X_MAX,
                 step =STEP,
                 n = N,
                 ds_type=DatasetType.UNIFORM,
                 coeff=COEFF,
                 c_min=C_MIN,
                 c_max=C_MAX
                 )

if config.mutation_type == MutationType.SAM:
    appendix = "sam"
else:
    appendix = "prob"

X,y = tiny_sr.generate_dataset(deg=D)
problem = BlackBox(observations_=X,
                   actual_=y,
                   loss_=mean_squared_error,
                   ideal_=IDEAL,
                   minimizing_=True)

config.ideal_fitness = problem.ideal
config.global_seed = int(time.time_ns())
cgp = SimpleCGP(functions, terminals, config, hyperparameters)

t0 = time.time()
best = cgp.evolve(problem)
t1 = time.time()
delta = t1 - t0

print(f"{D},simple_cgp_log_{appendix},{cgp.generation_number},{delta}")