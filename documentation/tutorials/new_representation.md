# Adding a representation module

This tutorial demonstrates the implementation of a new representation module based on the already existing TinyCGP module for Cartesian Genetic Programming (CGP).

## Create a new script
To add a new representation module, create a new Python file in the `src/gp` folder. The naming convention is `tiny_<acronym of representation>.py`.

**Example**: For Cartesian Genetic Programming (CGP), the file name should be `tiny_cgp.py`.

Finally, add a short description of the module:

```python
"""
TinyCGP: A minimalistic implementation of Cartesian Genetic Programming for
         TinyverseGP.
"""
```

## Implement the classes
Base classes that serve as a framework for the representation modules are provided in `src/gp/tinyverse` and need to be imported first:

```python
from gp.tinyverse import GPModel, GPConfig, GPIndividual, Hyperparameters
```

!!! Note
    Each class and method should contain docstrings in the Sphinx format.

### GPConfig
The configuration class is required for representation-specific parameters. For Cartesian Genetic Programming, these include the number of inputs and the number of outputs. The naming convention is `<acronym of representation>Config`. An example from the TinyCGP module is shown in the following:

```python
@dataclass(kw_only=True)
class CGPConfig(GPConfig):
    """
    Specialized GP configuration that is needed to run CGP.
    """

    num_inputs: int
    num_outputs: int
    num_functions: int
    max_arity: int
    max_time: int
    report_every_improvement: bool = False

    def __post_init__(self):
        self.genes_per_node = self.max_arity + 1
```

### GPIndividual
The naming convention for the class that implements `GPIndividual` is `<acronym of representation>Individual`. Individuals consist of their genome, for CGP that is a list of `int`s, and their fitness. `GPIndividual` further provides abstract methods `serialize_genome` and `deserialize_genome`. For CGP, the genome is already in a suitable serialisation format and can therefore be returned without change:

```python
def serialize_genome(self):
    return self.genome

def deserialize_genome(self, genome_):
    self.genome = genome_
```

### Hyperparameters
The `Hyperparameters` class manages the hyperparameters of a representation. The configuration space for the hyperparameters can also be specified here which is important for [hyperparameter optimisation with SMAC3](hpo_interface_tutorial.md). The naming convention is `<acronym of representation>Hyperparameters`.

```python
@dataclass(kw_only=True)
class CGPHyperparameters(Hyperparameters):
    """
    Specialized hyperparameter configuration space for CGP.
    """

    mu: int
    lmbda: int
    num_function_nodes: int
    strict_selection: bool
    levels_back: int
    mutation_rate: float
    population_size: Optional[int] = None
    mutation_rate_genes: int = None


    def __post_init__(self):
        Hyperparameters.__post_init__(self)
        self.space["mu"] = (1, 128)
        self.space["lmbda"] = (1, 4096)
        self.space["num_function_nodes"] = (10, 10000)
        self.space["levels_back"] = (1, 10000)
        self.space["strict_selection"] = [True, False]
        self.space["mutation_rate"] = (0.0, 1.0)
        self.space["penalization_complexity_factor"] = (0.0, 1.0)
        self.space["penalization_feasibility_factor"] = (0.0, 1.0)
        self.space["penalization_validity_factor"] = (0.0, 1.0)
        self.population_size = self.mu + self.lmbda
```

### GPModel
`Tiny<acronym of representation>` inherits from `GPModel` and is the main class of the module. The following methods should be implemented.

#### `init_population(self) -> list`
This function initialises individuals for the first generation of the evolutionary run. Initialisation differs from representation to representation. Tree-based GP, for example, commonly uses the Ramped Half-and-Half algorithm to initialise its population. For CGP, we simply append `hyperparameters.population_size` individuals to the `population`. A `population` is a list with elements of type `GPIndividual`.

```python
def init_population(self) -> list:
        """
        Initialization routine that creates and inits
        the individuals for the first generation.

        :return: list of individuals
        """
        self.population.clear()
        for _ in range(self.hyperparameters.population_size):
            individual = self.init_individual()
            self.population.append(individual)
```

#### `is_valid(self, genome: list[int])`
This function should check whether a given genome is valid.

#### `eval_complexity(self, genome: list[int]) -> float`
This function should return a value that approximates the complexity of evaluation for a given genome. For CGP, this approximation can be based on the number of active nodes in a genome:

```python
def eval_complexity(self, genome: list[int]) -> float:
        """
        Returns the complexity of the genome based on the number of active nodes.

        :param genome: Genome of an individual
        :return: Complexity value
        """
        active_nodes = self.active_nodes(genome)
        return len(active_nodes)
```

#### `evaluate_individual(self, genome: list[int], problem) -> float`
The fitness of an individual is calculated differently depending on the problem. The evaluation is therefore implemented in the according [`problem` module](new_problem_domain.md). `evaluate_individual` usually simply passes the genome of the individual to the `evaluate` method of the `problem` class:

```python
return problem.evaluate(genome, self)
```

#### `selection(self) -> list`
The selection of parent individuals should be carried out according to the chosen evolutionary strategy.


```python
def selection(self) -> list:
        """
        Performs a 1 + lambda strategy with either
        strict or non-strict selection. Non-strict selection
        allows to explore the neutral neighbourhood of the
        parent which has been found to be very effective for
        the use of CGP:

        :return: parent individual
        """
        sorted_pop = sorted(
            self.population,
            key=lambda ind: ind.fitness,
            reverse=not self.config.minimizing_fitness,
        )
        count = 0
        if not self.hyperparameters.strict_selection:
            best_fitness = sorted_pop[0].fitness
            for individual in sorted_pop:
                if individual.fitness != best_fitness:
                    break
                else:
                    count += 1
            parent = random.randint(0, count - 1)
        else:
            parent = 0
        return sorted_pop[parent]
```

#### `pipeline(self, problem)`
Implement one generational step. For most representations, this usually involves the `breed` method which determines the next generation of individuals based on the individuals of the current generation.

```python
def breed(self, parent: CGPIndividual = None):
        """
        Breeds lambda individuals with point mutation
        and adds them to the population.

        The genome is cloned before the mutation so that is
        guaranteed that the mutation is performed on the offspring's
        genome.

        :param parent: Individual selected to be the parent
        """
        self.population.clear()
        self.population.append(parent)
        for _ in range(self.hyperparameters.lmbda):
            offspring = CGPIndividual(parent.genome.copy())
            self.mutation(offspring.genome)
            self.population.append(offspring)
```

```python
def pipeline(self, problem):
        """
        Pipeline that performs one generational step CGP in the common
        1+lambda fashion.

        :return: best solution found in the population
        """

        parent = self.best_individual

        if not self.hyperparameters.strict_selection:
            parent = self.selection()

        # Population breeding
        self.breed(parent)

        # Evaluation of the offspring
        return self.evaluate(problem)
```

#### `predict(self, genome: list[int], observation: list) -> list`
`predict` computes the output of an individual based on its `genome` for a given `observation`. For representations such as CGP where multiple output nodes exist, `predict` returns a list.

```python
def predict(self, genome: list[int], observation: list) -> list:
        """
        Makes prediction based on a given observation and the paths
        of the decoded individual.

        Decodes the paths that are encoded in the genome if not obtained already
        to avoid unnecessary iterations.

        Makes use the memoization to avoid re-evaluation of function nodes.

        :param genome: The genome of the individual
        :param observation: Current observation
        :return: set of predicted values
        """
        node_map = dict()
        prediction = []

        self.current_paths = self.decode_optimized(genome)

        for path in self.current_paths:
            cost = 0.0
            for node_num in path:
                if node_num not in node_map.keys():
                    if node_num < self.config.num_inputs:
                        if self.terminals[node_num].const:
                            node_map[node_num] = self.terminals[node_num]()
                        else:
                            node_map[node_num] = observation[self.terminals[node_num]()]
                    else:
                        node_pos = self.node_position(node_num)
                        function = genome[node_pos]
                        connections = [
                            gene
                            for gene in genome[
                                node_pos
                                + 1 : node_pos
                                + self.function2arity[function]
                                + 1
                            ]
                        ]
                        args = [node_map[connection] for connection in connections]
                        node_map[node_num] = self.functions[function](*args)
                cost = node_map[node_num]
            prediction.append(cost)
        return prediction
```

#### `expression`
This function generates a human-readable representation of an individual based on its `genome`.

```python
def expression(self, genome: list[int]) -> list[str]:
        """
        Generates the symbolic expression by decoding all paths and storing the subexpressions
        of the respective function nodes in a map that is used to compose a human-readable form.

        :param genome: Genome of the individual that should be decoded to human readable form
        :return: list of expressions
        """

        def generate_expr_map(genome: list[int], active_nodes=None) -> dict:
            """
            Generates a map of subexpressions for each node that is later used
            to compose the symbolic expression.

            :param genome: Genome of the individual
            :param active_nodes: List of active function nodes
            :return: map with node subexpressions
            """
            if active_nodes is None:
                active_nodes = self.active_nodes(genome)
            expr_map = dict()
            for node_num in active_nodes:
                function = self.node_function(node_num, genome)
                node_arity = self.function2arity[function]
                args = self.node_connections(node_num, genome)[0:node_arity]
                func_name = self.functions[function].name
                node_expr = func_name + "("
                for index, argument in enumerate(args):
                    if argument in expr_map:
                        arg_expr = expr_map[argument]
                    else:
                        arg_expr = self.input_name(argument)
                    node_expr += arg_expr
                    if index < node_arity - 1:
                        node_expr += ", "
                node_expr += ")"
                expr_map[node_num] = node_expr
            return expr_map

        expr_map = generate_expr_map(genome)
        expressions = []

        outputs = self.get_outputs(genome)

        for output in outputs:
            if output < self.config.num_inputs:
                expression = self.input_name(output)
            else:
                expression = expr_map[output]
            expressions.append(expression)

        return expressions
```
