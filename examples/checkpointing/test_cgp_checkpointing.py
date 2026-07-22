from src.gp.tiny_cgp import TinyCGP, CGPConfig, CGPHyperparameters
from src.gp.problem import BlackBox
from src.benchmark.symbolic_regression.sr_benchmark import SRBenchmark
from src.gp.functions import ADD, SUB, MUL, DIV
from src.gp.loss import absolute_distance
from src.gp.tinyverse import Var, Const, Checkpointer

functions = [ADD, SUB, MUL, DIV]
terminals = [Var(0), Const(1)]

config = CGPConfig(
    global_seed=13,
    num_jobs=1,
    max_generations=200,
    stopping_criteria=1e-6,
    minimizing_fitness=True,
    ideal_fitness=1e-6,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=1,
    num_outputs=1,
    num_function_nodes=10,
    report_interval=100,
    max_time=60,
    checkpointing = True,
    checkpoint_interval=100,
    checkpoint_dir="checkpoints",
    experiment_name="my_experiment",
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=32,
    population_size=33,
    levels_back=len(terminals),
    mutation_rate=0.1,
    strict_selection=True,
)

loss = absolute_distance
benchmark = SRBenchmark()
data, actual = benchmark.generate("KOZA3")

problem = BlackBox(data, actual, loss, config.ideal_fitness, config.minimizing_fitness)

cgp = TinyCGP(functions, terminals, config, hyperparameters)
cgp.evolve(problem)

cp = Checkpointer(config, hyperparameters)
checkpoint_file = "checkpoints/my_experiment/checkpoint_gen_100.dill"
checkpoint = cp.load("checkpoints/my_experiment/checkpoint_gen_100.dill")

hyperparameters = CGPHyperparameters.from_dict(checkpoint["hyperparameters"])
config = CGPConfig.from_dict(checkpoint["config"])

cgp = TinyCGP(functions, terminals, config, hyperparameters)
cgp.resume(checkpoint, problem)
