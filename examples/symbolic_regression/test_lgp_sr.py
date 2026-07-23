"""
Example module to test LGP with symbolic regression problems.

Attempts to evolve a solution for the Koza-1 benchmkark which is
a quartic polynomial: x^4 + x^3 + x^2 + x

The problem is specified in the following paper:
https://dl.acm.org/doi/10.1145/2330163.2330273

Please note: This benchmark is nowadays considered a toy problem and
no serious benchmark. It only serves as an example for SR as an application
domain for TinyverseGP:
"""

from src.gp.tiny_lgp import TinyLGP, LGPConfig, LGPHyperparameters
from src.gp.problem import BlackBox
from src.benchmark.symbolic_regression.sr_benchmark import SRBenchmark
from src.gp.functions import ADD, SUB, MUL, DIV
from src.gp.loss import mean_squared_error
from src.gp.tinyverse import Var, Const

try:
    from icecream import ic
except ModuleNotFoundError:
    def ic (*args, **kwargs):
        return None

ARITHMETIC_FUNCTIONS = [ADD, SUB, MUL, DIV]


def main():
    functions = ARITHMETIC_FUNCTIONS
    terminals = [Var(0)]*4 + [Const(x) for x in range(1,7)]

    hyperparameters = LGPHyperparameters(
        mu=2000,
        macro_variation_rate=0.75,
        micro_variation_rate=0.25,
        insertion_rate=0.5,
        max_segment=15,
        reproduction_rate=0.5,
        branch_probability=0.0,
        p_register = 0.5,
        max_len = 200,
        initial_max_len = 35,
        erc = False,
        default_value = 0.0,
        protection = 1e10,
        penalization_validity_factor=0.0
    )
    config = LGPConfig(
        num_jobs=1,
        max_generations=500_000 - hyperparameters.mu,
        stopping_criteria=1e-6,
        minimizing_fitness=True,
        ideal_fitness=1e-6,
        silent_algorithm=False,
        silent_evolver=False,
        minimalistic_output=True,
        report_interval=100000000000,
        max_time=500,
        num_registers=8,
        global_seed=None,
        checkpoint_interval=1000000000000,
        checkpoint_dir="checkpoints",
        experiment_name="my_experiment",
    )

    loss = mean_squared_error
    data, actual = SRBenchmark().generate("KOZA3")
    problem = BlackBox(data, actual, loss, 1e-6, True)

    lgp = TinyLGP(functions, terminals, config, hyperparameters)
    lgp.evolve(problem)
    #print(lgp.expression(lgp.best_individual.genome))


if __name__ == "__main__":
    main()
