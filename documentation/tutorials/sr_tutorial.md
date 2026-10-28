# Running a Symbolic Regression (SR) experiment

This tutorial is based on `examples/symbolic_regression/test_tgp_sr.py`, `examples/symbolic_regression/test_cgp_sr.py`, `examples/symbolic_regression/test_lgp_sr.py`, and `examples/symbolic_regression/test_ge_sr.py` and demonstrates how to perform an experiment using Tree-based Genetic Programming (TGP), Cartesian Genetic Programming (CGP), Linear Genetic Programming (LGP), or Grammatical Evolution (GE) on KOZA symbolic regression problems.
    
## 1. Imports

First, set up the necessary imports. `tiny_[tgp|cgp|lgp|ge]` is the module for either TGP, CGP, LGP, or GE, respectively. `problem` and `sr_benchmark` are required to formulate the problem, a collection of commonly used functions are implemented in `functions`, `loss` contains common loss functions, and `tinyverse` contains a collection of base classes for the GP modules.

=== "TGP"

    ```python
    from gp.tiny_tgp import TinyTGP, TGPHyperparameters
    from gp.functions import ADD, SUB, MUL, DIV
    from gp.loss import absolute_distance
    from gp.problem import BlackBox
    from benchmark.symbolic_regression.sr_benchmark import SRBenchmark
    from gp.tinyverse import GPConfig, Var, Const
    ```

=== "CGP"

    ```python
    from gp.tiny_cgp import TinyCGP, CGPConfig, CGPHyperparameters
    from gp.problem import BlackBox
    from benchmark.symbolic_regression.sr_benchmark import SRBenchmark
    from gp.functions import ADD, SUB, MUL, DIV
    from gp.loss import absolute_distance
    from gp.tinyverse import Var, Const
    ```
    
=== "LGP"

    ```python
    from gp.tiny_lgp import TinyLGP, LGPConfig, LGPHyperparameters
    from gp.problem import BlackBox
    from benchmark.symbolic_regression.sr_benchmark import SRBenchmark
    from gp.functions import ADD, SUB, MUL, DIV
    from gp.loss import mean_squared_error
    from gp.tinyverse import Var, Const
    ```

=== "GE"

    ```python
    from gp.tiny_ge import TinyGE, GEHyperparameters
    from gp.functions import ADD, SUB, MUL, DIV
    from gp.loss import absolute_distance
    from gp.problem import BlackBox
    from benchmark.symbolic_regression.sr_benchmark import SRBenchmark
    from gp.tinyverse import GPConfig
    ```

For TGP and GE, we additionally import `GPConfig` base class from `tinyverse` as they do not implement their own configuration classes. GE does not use terminals but arguments and importing `Var` and `Const` for the variables and constants is therefore not necessary.

## 2. Choose the function and terminal set

Functions are implemented in `src/gp/functions.py`. For the terminals, use the `Var` class for variables and `Const` class for constants.

=== "TGP"

    ```python title="Example"
    functions = [ADD, SUB, MUL, DIV]
    terminals = [Var(0), Const(1)]
    ```
    
=== "CGP"

    ```python title="Example"
    functions = [ADD, SUB, MUL, DIV]
    terminals = [Var(0), Const(1)]
    ```
    
=== "LGP"

    ```python title="Example"
    functions = [ADD, SUB, MUL, DIV]
    terminals = [Var(0)]*4 + [Const(x) for x in range(1,7)]
    ```

### GE: Functions, arguments and grammar

GE uses a function set but no terminal set. Instead, arguments and the grammar need to be set:

```python title="Example"
functions = [ADD, SUB, MUL, DIV]
arguments = ["x"]
grammar = {
    "<expr>": [
        "ADD(<expr>, <expr>)",
        "SUB(<expr>, <expr>)",
        "MUL(<expr>, <expr>)",
        "DIV(<expr>, <expr>)",
        "<d>",
        "<d>.<d><d>",
        "x",
    ],
    "<d>": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
}
```

## 3. Configure the run and model

`GPConfig` is used to configure the evolutionary run as well as settings specific to the according representation. CGP and LGP implement their own configuration classes.

=== "TGP"

    ```python title="Example"
    config = GPConfig(
        num_jobs=1,
        max_generations=100,
        stopping_criteria=1e-6,
        minimizing_fitness=True,
        ideal_fitness=1e-6,
        silent_algorithm=False,
        silent_evolver=False,
        minimalistic_output=False,
        num_outputs=1,
        report_interval=1,
        max_time=60,
        constraints = lambda x: max(0, number_divs(x) - 1),
        global_seed=42,
        checkpoint_interval=10,
        checkpoint_dir='checkpoint',
        experiment_name='sr_tgp'
    )
    ```

=== "CGP"

    ```python title="Example"
    config = CGPConfig(
        num_jobs=1,
        max_generations=100,
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
        report_interval=1,
        max_time=60,
        global_seed=42,
        checkpoint_interval=10,
        checkpoint_dir='checkpoint',
        experiment_name='sr_cgp'
    )
    ```

=== "LGP"

    ```python title="Example"
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
        experiment_name="sr_lgp",
    )
    ```

=== "GE"

    ```python title="Example"
    config = GPConfig(
        num_jobs=1,
        max_generations=100,
        stopping_criteria=1e-6,
        minimizing_fitness=True,
        ideal_fitness=1e-6,
        silent_algorithm=False,
        silent_evolver=False,
        minimalistic_output=True,
        num_outputs=1,
        report_interval=1,
        max_time=60,
        global_seed=42,
        checkpoint_interval=10,
        checkpoint_dir='examples/checkpoint',
        experiment_name='sr_ge'
    )
    ```

For TGP, we use `number_divs` and `count_divs` which are defined as follows:

```python
def number_divs(individual):
    """Count the number of divisions in an individual."""
    return len([count_divs(node) for node in individual])

def count_divs(node):
    """Count the number of divisions in an individual."""
    if node.function == DIV:
        return 1 + sum(count_divs(child) for child in node.children)
    return sum(count_divs(child) for child in node.children)
```

## 4. Set the hyperparameters

Hyperparameters can be set using the according `GPHyperparameters` class.

=== "TGP"

    ```python title="Example"
    hyperparameters = TGPHyperparameters(
        pop_size=100,
        max_size=25,
        max_depth=5,
        cx_rate=0.9,
        mutation_rate=0.3,
        tournament_size=2,
        penalization_complexity_factor=0.1,
        erc=False
    )
    ```

=== "CGP"

    ```python title="Example"
    hyperparameters = CGPHyperparameters(
        mu=1,
        lmbda=32,
        population_size=33,
        num_function_nodes=10,
        levels_back=len(terminals),
        mutation_rate=0.1,
        strict_selection=True,
    )
    ```

=== "LGP"

    ```python title="Example"
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
        penalty_value=99999,
    )
    ```

## 5. Configure the experiment

Now select the loss function and problem domain as well as the problem instance you want to run the experiment on. The data is generated via the `SRBenchmark` class and is then used for the problem definition with `BlackBox`.

```python title="Example"
loss = absolute_distance
benchmark = SRBenchmark()
data, actual = benchmark.generate("KOZA3")

problem = BlackBox(data, actual, loss, 1e-6, True)
```

## 6. Start the evolutionary run

Instantiate the model and start the evolutionary run using `evolve()`.

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

Name your file (for example, `test_cgp_sr.py`) and run the experiment.

```bash title="Example"
python3 test_cgp_sr.py
```

If `silent_algorithm`, `silent_evolver`, and `minimalistic_output` are set to `False`, the output should look similar to this:

```bash
...
Generation #47 - Best Fitness: 87163.22682770062
Generation #48 - Best Fitness: 4.064325473396302e-10
Ideal fitness found in generation 48
Job #0 - Evaluations: 1650 - Best Fitness: 4.064325473396302e-10
```

For each generation, the fitness of the best individual is printed. At the end of the run, the job number and the best fitness of the run are reported.

!!! Note 
    Output may vary depending on whether an individual with ideal fitness was found, whether the maximum number of generations was reached, whether the time limit has been exceeded, and the configuration of `silent_algorithm`, `silent_evolver`, and `minimalistic_output`.
