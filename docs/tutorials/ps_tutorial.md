# Program Synthesis (PS)

This tutorial is based on `examples/program_synthesis/test_tgp_ps.py`, `examples/program_synthesis/test_cgp_ps.py`, and `examples/program_synthesis/test_ge_ps.py` and demonstrates how to use Tree-based Genetic Programming (TGP), Cartesian Genetic Programming (CGP), or Grammatical Evolution (GE) to perform program synthesis experiments. In this tutorial, we attempt to evolve a solution for the [Power of Two Leetcode problem](https://leetcode.com/problems/power-of-two/description/).

## 1. Imports

First, set up the necessary imports. `tiny_[tgp|cgp|ge]` is the module for either TGP, CGP, or GE, respectively. `problem`, `ps_benchmark`, and `power_of_two` are required to formulate the problem. A collection of commonly used functions are implemented in `functions`, `loss` contains common loss functions, and `tinyverse` contains a collection of base classes for the GP modules.

=== "TGP"

    ```python
    from gp.tinyverse import GPConfig
    from gp.problem import ProgramSynthesis
    from benchmark.program_synthesis.ps_benchmark import PSBenchmark
    from benchmark.program_synthesis.leetcode.power_of_two import gen_power_of_two
    from gp.functions import ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, IF, LT, GT
    from gp.tiny_tgp import TinyTGP, TGPHyperparameters
    from gp.tinyverse import Var, Const
    ```

=== "CGP"

    ```python
    from gp.tiny_cgp import *
    from gp.problem import ProgramSynthesis
    from benchmark.program_synthesis.ps_benchmark import PSBenchmark
    from benchmark.program_synthesis.leetcode.power_of_two import *
    from gp.functions import *
    from gp.tinyverse import Var, Const
    ```

=== "GE"

    ```python
    from gp.tinyverse import GPConfig
    from gp.problem import ProgramSynthesis
    from benchmark.program_synthesis.ps_benchmark import PSBenchmark
    from benchmark.program_synthesis.leetcode.power_of_two import gen_power_of_two
    from gp.functions import ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, IF, LT, GT
    from gp.tiny_ge import TinyGE, GEHyperparameters
    ```

## 2. Set the function and terminal set

Functions are implemented in `src/gp/functions.py`. For the terminals, use the `Var` class for variables and `Const` class for constants.

```python title="Example"
NUM_INPUTS = 1
functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, IF, LT, GT]
terminals = [Var(i) for i in range(NUM_INPUTS)] + [Const(1), Const(2)]
```

### GE: Functions, arguments and grammar

GE uses a function set but no terminal set. Instead, arguments and the grammar need to be set:

```python title="Example"
functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, IF, LT, GT]
arguments = ["x"]
grammar = {
    "<expr>": [
        "ADD(<expr>, <expr>)",
        "SUB(<expr>, <expr>)",
        "MUL(<expr>, <expr>)",
        "DIV(<expr>, <expr>)",
        "AND(<expr>, <expr>)",
        "OR(<expr>, <expr>)",
        "NAND(<expr>, <expr>)",
        "NOR(<expr>, <expr>)",
        "NOT(<expr>)",
        "IF(<expr>, <expr>, <expr>)",
        "LT(<expr>, <expr>)",
        "GT(<expr>, <expr>)",
        "<d>",
        "<d>.<d><d>",
        "x",
    ],
    "<d>": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
}
```

## 3. Configure the run and model

=== "TGP"

    ```python title="Example"
    config = GPConfig(
        global_seed=42,
        num_jobs=1,
        max_generations=1000,
        stopping_criteria=100,
        minimizing_fitness=False,
        ideal_fitness=100,
        silent_algorithm=False,
        silent_evolver=False,
        minimalistic_output=True,
        num_outputs=1,
        report_interval=1,
        max_time=60,
        checkpoint_dir='checkpoint',
    )
    ```

=== "CGP"

    ```python title="Example"
    config = CGPConfig(
        num_jobs=1,
        max_generations=10000,
        stopping_criteria=100,
        minimizing_fitness=False,
        ideal_fitness=100,
        silent_algorithm=False,
        silent_evolver=False,
        minimalistic_output=True,
        num_functions=len(functions),
        max_arity=3,
        num_inputs=NUM_INPUTS,
        num_outputs=1,
        global_seed=42,
        report_interval=1,
        max_time=60,
        checkpoint_interval=10,
        checkpoint_dir='checkpoint',
        experiment_name='ps_cgp'
    )
    ```

=== "GE"

    ```python title="Example"
    config = GPConfig(
        num_jobs=1,
        max_generations=1000,
        stopping_criteria=100,
        minimizing_fitness=False,
        ideal_fitness=100,
        silent_algorithm=False,
        silent_evolver=False,
        minimalistic_output=True,
        num_outputs=1,
        global_seed=42,
        report_interval=1,
        max_time=60,
        checkpoint_dir='checkpoint',
    )
    ```

## 4. Set the hyperparameters

=== "TGP"

    ```python title="Example"
    hyperparameters = TGPHyperparameters(
        pop_size=100,
        max_size=25,
        max_depth=5,
        cx_rate=0.9,
        mutation_rate=0.3,
        tournament_size=2,
        erc=True
    )
    ```

=== "CGP"

    ```python title="Example"
    hyperparameters = CGPHyperparameters(
        mu=1,
        lmbda=32,
        population_size=33,
        num_function_nodes=100,
        levels_back=len(terminals),
        mutation_rate=0.1,
        strict_selection=True,
    )
    ```

=== "GE"

    ```python title="Example"
    hyperparameters = GEHyperparameters(
        pop_size=100,
        genome_length=40,
        codon_size=1000,
        cx_rate=0.9,
        mutation_rate=0.1,
        tournament_size=2,
        penalty_value=0,
    )
    ```

## 5. Define the problem

```python title="Example"
generator = gen_power_of_two
n = 10
m = 100

benchmark = PSBenchmark(generator, [n, m])
problem = ProgramSynthesis(benchmark.dataset)
```

## 6. Evolve a solution

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

=== "GE"

    ```python
    ge = TinyGE(functions, grammar, arguments, config, hyperparameters)
    ge.evolve(problem)
    ```

## Run it

Name your file (for example, `test_tgp_ps.py`) and run the experiment.

```bash title="Example"
python3 test_tgp_ps.py
```

If `silent_algorithm` and `silent_evolver` are set to `False`, the output should look similar to this:

```bash
...
Generation #450 - Best Fitness: 100.0
Generation #451 - Best Fitness: 101.0
Ideal fitness found in generation 451
45300;101.0
```

For each generation, the fitness of the best individual is printed. At the end of the run, the number of evaluations and the best fitness of the run are reported.

!!! Note 
    Output may vary depending on whether an individual with ideal fitness was found, whether the maximum number of generations was reached, whether the time limit has been exceeded, and the configuration of `silent_algorithm`, `silent_evolver`, and `minimalistic_output`.
