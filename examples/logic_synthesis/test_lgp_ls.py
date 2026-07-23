"""
Example module to test TGP with logic synthesis problems.

Evolves a symbolic expression for a boolean function that
is presented as a compressed or uncompressed truth table.

The logic synthesis benchmark files are located in the data folder.

The provided benchmarks are taken from the General Boolean Function Benchmark Suite (GFBS):

https://dl.acm.org/doi/10.1145/3594805.3607131
"""
from src.gp.tinyverse import Var
from src.benchmark.logic_synthesis.ls_benchmark import LSBenchmark
from src.gp.tiny_lgp import TinyLGP, LGPConfig, LGPHyperparameters
from src.gp.functions import AND, OR, NAND, NOR, XOR, NOT
from src.gp.loss import hamming_distance_bitwise
from src.gp.problem import BlackBox

benchmark = LSBenchmark('data/logic_synthesis/plu/add3.plu')
benchmark.generate()
truth_table = benchmark.get_truth_table()
num_inputs = benchmark.benchmark.num_inputs
num_outputs = benchmark.benchmark.num_outputs
print(num_inputs, num_outputs)

functions = [AND, OR, NAND, NOR, XOR, NOT]
terminals = [Var(i) for i in range(num_inputs)]

hyperparameters = LGPHyperparameters(
    mu=5000,
    macro_variation_rate=0.75,
    micro_variation_rate=0.25,
    insertion_rate=0.4,
    max_segment=15,
    reproduction_rate=0.2,
    branch_probability=0.0,
    p_register = 0.1,
    max_len = 150,
    initial_max_len = 35,
    erc = False,
    default_value = 0,
    protection = 1e10,
    penalization_validity_factor=0.0
)
config = LGPConfig(
        num_jobs=1,
        max_generations=1_000_000, # - hyperparameters.mu,
        stopping_criteria=0,
        minimizing_fitness=True,
        ideal_fitness=0,
        silent_algorithm=False,
        silent_evolver=False,
        minimalistic_output=True,
        report_interval=10*hyperparameters.mu,
        max_time=2000,
        num_outputs=num_outputs,
        num_registers=6,
        global_seed=None,
        checkpoint_interval=1000000000,
        checkpoint_dir="checkpoints",
        experiment_name="logic_lgp",
)

loss = hamming_distance_bitwise
data = truth_table.inputs
actual = truth_table.outputs
problem = BlackBox(data, actual, loss, 0, True)

lgp = TinyLGP(functions, terminals, config, hyperparameters)
lgp.evolve(problem)
print(lgp.expression(lgp.best_individual.genome))
