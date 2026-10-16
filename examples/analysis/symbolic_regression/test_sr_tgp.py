import sys
import time
from src.analysis.benchmarks.symbolic_regression import TinySR
from src.analysis.models.simple_tgp import SimpleTGPConfig, InitMethod, SimpleTGP, MutationType, SimpleTGPHyperparameters
from src.gp.functions import ADD, MUL
from src.gp.loss import absolute_distance
from src.gp.problem import BlackBox
from src.gp.tinyverse import Var

MAX_GENERATIONS = 2000000
MAX_TIME = 999999
D = int(sys.argv[1])
functions = [ADD, MUL]
terminals = [Var(0)]
X_MIN = -1.0
X_MAX = 1.0
N = 20
IDEAL = 1e-3
MAX_DEPTH = D

config = SimpleTGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=IDEAL,
    minimizing_fitness=True,
    ideal_fitness=IDEAL,
    init_method=InitMethod.MIN,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1000,
    max_time=MAX_TIME,
    global_seed=None,
    checkpoint_interval=9999999,
    checkpoint_dir='../checkpoint',
    experiment_name='tiny_sr_tgp'
)

hyperparameters = SimpleTGPHyperparameters(
    lmbda=1,
    k=1,
    strict_selection=False,
    check_complexity=False,
    min_depth=1,
    max_depth=D,
    multi=True,
    discard_invalid=True,
    mutation_type=MutationType.HVL_STD
)

tiny_sr = TinySR(x_min=X_MIN,
                 x_max=X_MAX,
                 n=N)

match config.init_method:
    case InitMethod.GROW:
        init_appendix = "grow"
    case InitMethod.FULL:
        init_appendix = "full"
    case InitMethod.MIN:
        init_appendix = "min"

match hyperparameters.mutation_type:
    case MutationType.HVL_DEPTH_UNBIASED:
        hvl_appendix1 = "depth_unbiased"
    case MutationType.HVL_NODE_UNBIASED:
        hvl_appendix1 = "node_unbiased"
    case MutationType.HVL_STD:
        hvl_appendix1 = "std"

if hyperparameters.multi:
    hvl_appendix2 = "multi"
else:
    hvl_appendix2 = "single"


X,y = tiny_sr.generate_dataset(deg=D)
problem = BlackBox(observations_=X,
                   actual_=y,
                   loss_=absolute_distance,
                   ideal_=IDEAL,
                   minimizing_=True)

config.ideal_fitness = problem.ideal
config.global_seed = int(time.time_ns())
tgp = SimpleTGP(functions, terminals, config, hyperparameters)

t0 = time.time()
best = tgp.evolve(problem)
t1 = time.time()
delta = t1 - t0

print(f"{D},simple_tgp_{hvl_appendix1}_{hvl_appendix2}_{init_appendix},{tgp.generation_number},{delta}, {best.fitness}")

