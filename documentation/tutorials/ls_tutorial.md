# Running a Logic Synthesis (LS) experiment

This tutorial is based on `examples/logic_synthesis/test_tgp_ls.py`, `examples/logic_synthesis/test_cgp_ls.py`, `examples/logic_synthesis/test_lgp_ls.py`, and `examples/logic_synthesis/test_ge_ls.py` and demonstrates how to perform an experiment using Tree-based Genetic Programming (TGP), Cartesian Genetic Programming (CGP), Linear Genetic Programming (LGP), or Grammatical Evolution (GE) on logic synthesis benchmarks from the [General Boolean Function Benchmark Suite (GBFS)](https://dl.acm.org/doi/10.1145/3594805.3607131).

## 1. Imports

First, set up the necessary imports. `tiny_[tgp|cgp|lgp|ge]` is the module for either TGP, CGP, LGP, or GE, respectively. `problem` and `ls_benchmark` are required to formulate the problem, a collection of commonly used functions are implemented in `functions`, `loss` contains common loss functions, and `tinyverse` contains a collection of base classes for the GP modules.

=== "TGP"

    ```python
    from gp.tinyverse import Var
    from benchmark.logic_synthesis.ls_benchmark import LSBenchmark
    from gp.tiny_tgp import TinyTGP, TGPHyperparameters, GPConfig
    from gp.functions import AND, OR, NAND, NOR
    from gp.loss import hamming_distance_bitwise
    from gp.problem import BlackBox
    ```

=== "CGP"

    ```python
    from benchmark.logic_synthesis.ls_benchmark import LSBenchmark
    from gp.tiny_cgp import *
    from gp.problem import BlackBox
    from gp.functions import *
    from gp.loss import *
    from gp.tinyverse import Var
    ```

=== "LGP"

    ```python
    from gp.tinyverse import Var
    from benchmark.logic_synthesis.ls_benchmark import LSBenchmark
    from gp.tiny_lgp import TinyLGP, LGPConfig, LGPHyperparameters
    from gp.functions import AND, OR, NAND, NOR, XOR, NOT
    from gp.loss import hamming_distance_bitwise
    from gp.problem import BlackBox
    ```

=== "GE"

    ```python
    from gp.tinyverse import GPConfig
    from benchmark.logic_synthesis.ls_benchmark import LSBenchmark
    from gp.tiny_ge import TinyGE, GEHyperparameters
    from gp.functions import AND, OR, NAND, NOR, XOR, NOT
    from gp.loss import hamming_distance_bitwise
    from gp.problem import BlackBox
    ```

## 2. Set up the benchmark

Choose the benchmark among the files in `data/logic_synthesis/plu/` or `data/logic_synthesis/tt/` and set the number of inputs and outputs.

```python title="Example"
benchmark = LSBenchmark('plu/add3.plu')
benchmark.generate()
truth_table = benchmark.get_truth_table()
num_inputs = benchmark.benchmark.num_inputs
num_outputs = benchmark.benchmark.num_outputs
```

## 3. Choose the function and terminal set

Set the function and terminal set. A number of functions are implemented in `src.gp.functions.py`. Use the `Var` class for variables.

=== "TGP"

    ```python title="Example"
    functions = [AND, OR, NAND, NOR]
    terminals = [Var(i) for i in range(num_inputs)]
    ```

=== "CGP"

    ```python title="Example"
    functions = [AND, OR, NAND, NOR]
    terminals = [Var(i) for i in range(num_inputs)]
    ```

=== "LGP"

    ```python title="Example"
    functions = [AND, OR, NAND, NOR, XOR, NOT]
    terminals = [Var(i) for i in range(num_inputs)]
    ```

### GE: Functions, arguments and grammar

GE uses a function set but no terminal set. Instead, arguments and the grammar need to be set:

```python title="Example"
functions = [AND, OR, NAND, NOR, XOR, NOT]
arguments = ["a", "b", "c", "d", "e", "f", "g"]
grammar = {
    "<expr>": ["[<lexpr>, <lexpr>, <lexpr>, <lexpr>]"],
    "<lexpr>": [
        "AND(<vexpr>, <vexpr>)",
        "OR(<vexpr>, <vexpr>)",
        "NAND(<vexpr>, <vexpr>)",
        "NOR(<vexpr>, <vexpr>)",
        "XOR(<vexpr>, <vexpr>)",
        "NOT(<vexpr>)",
    ],
    "<vexpr>": [
        "AND(<vexpr>, <vexpr>)",
        "OR(<vexpr>, <vexpr>)",
        "NAND(<vexpr>, <vexpr>)",
        "NOR(<vexpr>, <vexpr>)",
        "XOR(<vexpr>, <vexpr>)",
        "NOT(<vexpr>)",
        "<var>"
    ],
    "<var>": ["a", "b", "c", "d", "e", "f", "g"]
}
```

## 4. Configure the run and model

=== "TGP"

    ```python title="Example"
    config = GPConfig(
        num_jobs=1,
        max_generations=1000,
        stopping_criteria=1e-6,
        minimizing_fitness=True,
        ideal_fitness=1e-6,
        silent_algorithm=False,
        silent_evolver=False,
        minimalistic_output=True,
        num_outputs=num_outputs,
        report_interval=1,
        max_time=360,
        global_seed=42,
        checkpoint_interval=10,
        checkpoint_dir='examples/checkpoint',
        experiment_name='logic_tgp'
    )
    ```

=== "CGP"

    ```python title="Example"
    config = CGPConfig(
        num_jobs=1,
        max_generations=2000000,
        stopping_criteria=0,
        minimizing_fitness=True,
        ideal_fitness=0,
        silent_algorithm=False,
        silent_evolver=False,
        minimalistic_output=True,
        num_functions=len(functions),
        max_arity=2,
        num_inputs=len(terminals),
        num_outputs=num_outputs,
        report_interval=5000,
        max_time=360,
        global_seed=42,
        checkpoint_interval=10,
        checkpoint_dir='examples/checkpoint',
        experiment_name='logic_cgp'
    )
    ```

=== "LGP"

    ```python title="Example"
    config = LGPConfig(
        num_jobs=1,
        max_generations=1_000_000,
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
    ```

=== "GE"

    ```python title="Example"
    config = GPConfig(
        num_jobs=1,
        max_generations=50_000,
        stopping_criteria=1e-6,
        minimizing_fitness=True,
        ideal_fitness=0,
        silent_algorithm=False,
        silent_evolver=False,
        minimalistic_output=True,
        num_outputs=num_outputs,
        report_interval=100,
        max_time=500,
        global_seed=None,
        checkpoint_interval=100000,
        checkpoint_dir='examples/checkpoint',
        experiment_name='logic_ge'
    )
    ```

## 5. Set the hyperparameters

=== "TGP"

    ```python title="Example"
    hyperparameters = TGPHyperparameters(
        pop_size=2000,
        max_size=25,
        max_depth=5,
        cx_rate=0.9,
        mutation_rate=0.3,
        tournament_size=2,
        erc=False
    )
    ```

=== "CGP"

    ```python title="Example"
    hyperparameters = CGPHyperparameters(
        mu=1,
        lmbda=1,
        population_size=1000,
        levels_back=1000,
        mutation_rate=0.1,
        num_function_nodes=30,
        strict_selection=False,
    )
    ```

=== "LGP"

    ```python title="Example"
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
    ```

=== "GE"

    ```python title="Example"
    hyperparameters = GEHyperparameters(
        pop_size=1000,
        genome_length=60,
        codon_size=100,
        cx_rate=0.95,
        mutation_rate=0.25,
        tournament_size=3,
        penalty_value=99999,
    )
    ```

## 6. Configure the experiment

Define the problem.

```python title="Example"
loss = hamming_distance_bitwise
data = truth_table.inputs
actual = truth_table.outputs
problem = BlackBox(data, actual, loss, 0, True)
```

## 7. Start the evolutionary run

=== "TGP"

    ```python
    tgp = TinyTGP(functions, terminals, config, hyperparameters)
    tgp.evolve(problem)
    ```

=== "CGP"

    ```python
    cgp = TinyCGP(functions, terminals, config, hyperparameters)
    cgp.evolve(problem)
    ```

=== "LGP"

    ```python
    lgp = TinyLGP(functions, terminals, config, hyperparameters)
    lgp.evolve(problem)
    ```

=== "GE"

    ```python
    ge = TinyGE(functions, grammar, arguments, config, hyperparameters)
    ge.evolve(problem)
    ```

## Run the experiment 

Name your file (for example, `test_ge_ls.py`) and run the experiment.

```bash title="Example"
python3 test_ge_ls.py
```

If `silent_algorithm` and `silent_evolver` are set to `False`, the output should look similar to this:

```bash
...
Generation #3100 - Best Fitness: 416.0
Generation #3200 - Best Fitness: 416.0
Timelimit exceeded
3218000;416.0
```

For each generation, the fitness of the best individual is printed. At the end of the run, the number of evaluations and the best fitness of the run are reported.

!!! Note 
    Output may vary depending on whether an individual with ideal fitness was found, whether the maximum number of generations was reached, whether the time limit has been exceeded, and the configuration of `silent_algorithm`, `silent_evolver`, and `minimalistic_output`.
