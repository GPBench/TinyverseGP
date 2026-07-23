from src.gp.tiny_3ge import TreeGEHyperparameters
from src.gp.functions import ADD, SUB, MUL, DIV
from src.gp.loss import absolute_distance
from src.gp.problem import BlackBox
from src.benchmark.symbolic_regression.sr_benchmark import SRBenchmark
from src.gp.tiny_3ge import TreeGEConfig, Tiny3GE 
from src.hpo.hpo import SMACInterface



config = TreeGEConfig(
    num_jobs=1,
    max_generations=100,
    stopping_criteria=1e-6,
    minimizing_fitness=True,  # this should be used from the problem instance
    ideal_fitness=1e-6,  # this should be used from the problem instance
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1,
    max_time=60,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='sr_ge'
)

hyperparameters = TreeGEHyperparameters(
    pop_size=100,
    min_depth=4,
    max_depth=6,
    codon_size=256,
    cx_rate=0.9,
    mutation_rate=0.1,
    tournament_size=2,
    penalty_value=99999,
)

grammar = {
    "<expr>": [
        "ADD(<expr>, <expr>)",
        "SUB(<expr>, <expr>)",
        "MUL(<expr>, <expr>)",
        "DIV(<expr>, <expr>)",
        "<d>",
        "<d>.<d><d>",
        "<var>",
    ],
    "<d>": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    "<var>": ["x"],
}

loss = absolute_distance
benchmark = SRBenchmark()
data, actual = benchmark.generate("KOZA1")
functions = [ADD, SUB, MUL, DIV]
arguments = ["x"]
trials = 5 #20

problem = BlackBox(data, actual, loss, 1e-6, True)
ge = Tiny3GE(functions, grammar, arguments, config, hyperparameters)

interface = SMACInterface()

opt_hyperparameters = interface.optimise(ge, problem, trials)
print(opt_hyperparameters)

config.silent_algorithm=False
config.silent_evolver=False
ge = Tiny3GE(functions, grammar, arguments, config, opt_hyperparameters)
ge.evolve(problem)