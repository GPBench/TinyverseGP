"""
TinyCGP: A minimalistic implementation of Cartesian Genetic Programming for
         TinyverseGP.

         Genome representation: Standard integer-based CGP
         Mutation operator: Point mutation
         Search algorithm: 1+lambda ES with non-strict selection option
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.gp.tinyverse import (
    GPModel,
    GPConfig,
    Var,
    GPIndividual,
    Hyperparameters,
)


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

class CGPIndividual(GPIndividual):
    """
    Class that is used to represent a CGP individual.
    Formally a GP individual can be represented as a tuple consisting of
    the genome and the fitness value.

    Additionally, the CGP individual can store the path that are encoded in the genotype to
    avoid unnecessary evaluation costs by re-evaluating and re-visiting nodes in the
    decoding routine.
    """

    genome: list[int]
    fitness: any
    paths: list

    def __init__(self, genome_: list[int], fitness_: any = None, paths_=None):
        GPIndividual.__init__(self, genome_, fitness_)
        self.paths = paths_

    def __str__(self):
        return str(self.genome) + ";" + str(self.fitness)

    def serialize_genome(self):
        return self.genome

    def deserialize_genome(self, genome_):
        self.genome = genome_


class TinyCGP(GPModel):
    """
    Main class of the tiny CGP module that derives from GPModel and
    implements all related fundamental mechanisms tun run CGP.
    """

    class GeneType(Enum):
        """
        Enum for the gene type that are used for the CGP encoding.
        """

        FUNCTION = 0
        CONNECTION = 1
        OUTPUT = 2

    class TerminalType(Enum):
        """
        Enum used to specify the type of terminal symbols.
        """

        VARIABLE = 0
        CONSTANT = 1

    def __init__(
        self,
        functions_: list,
        terminals_: list,
        config_: CGPConfig,
        hyperparameters_: CGPHyperparameters,
    ):
        super().__init__(config_, hyperparameters_)

        self.config.num_genes = (
            self.config.genes_per_node * self.hyperparameters.num_function_nodes
        ) + self.config.num_outputs

        self.num_evaluations = 0
        self.population = []
        self.functions = functions_
        self.function2arity = [f.arity for f in functions_]
        self.terminals = terminals_
        self.inputs = dict()
        self.current_paths = None
        self.init_inputs(terminals_)
        self.init_population()

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

    def init_individual(self) -> CGPIndividual:
        """
        Creates and initializes an individual.

        :return CGP individual
        """
        return CGPIndividual(self.init_genome())

    def init_inputs(self, terminals_: list):
        """
        Initializes the inputs by taking the passed terminals.

        :param terminals_: Terminal symbols passed to the CGP class.
        """
        for index, terminal in enumerate(terminals_):
            if isinstance(terminal, Var):
                self.inputs[index] = (terminal, self.TerminalType.VARIABLE)
            else:
                self.inputs[index] = (terminal, self.TerminalType.CONSTANT)

    def init_genome(self) -> list[int]:
        """
        Initializes the genome by initializing each gene w.r.t. its type.

        :return: list of genes
        """
        return [self.init_gene(i) for i in range(self.config.num_genes)]

    def init_gene(self, position: int) -> int:
        """
        Initializes a gene at a specified position and returns the
        respective value.

        :param position: position of the gene in the genotype.
        :return: gene value
        """
        gene_type = self.phenotype(position)
        levels_back = self.hyperparameters.levels_back
        if gene_type == self.GeneType.CONNECTION:
            node_num = self.node_number(position)
            if node_num <= levels_back:
                return random.randint(0, node_num - 1)
            else:
                return random.randint(node_num - levels_back, node_num - 1)
        elif gene_type == self.GeneType.FUNCTION:
            return random.randint(0, self.config.num_functions - 1)
        else:
            rand = random.randint(
                0, self.config.num_inputs + self.hyperparameters.num_function_nodes - 1
            )
            return rand

    def phenotype(self, position: int) -> GeneType:
        """
        Return the phenotype of a gene.

        :param position: gene position in the genome
        :return: gene type
        """
        if position >= self.hyperparameters.num_function_nodes * (self.config.max_arity + 1):
            return self.GeneType.OUTPUT
        else:
            return (
                self.GeneType.FUNCTION
                if position % (self.config.max_arity + 1) == 0
                else self.GeneType.CONNECTION
            )

    def input_value(self, index: int) -> any:
        """
        Returns the input at a specified index.

        :param index: input index
        :return:input value
        """
        return self.inputs[index][0]

    def input_name(self, index: int) -> str:
        """
        Returns the name of an index at a specified index.

        :param index: input index
        :return:input name
        """
        idx = self.input_value(index)()
        return (
            f"{self.input_value(index).name}({idx})"
            if idx is not None
            else self.input_value(index).name
        )

    def input_type(self, index: int) -> TerminalType:
        """
        Returns the type of an index at a specified index.

        :param index: input index
        :return:input type
        """
        return self.inputs[index][1]

    def node_number(self, position: int) -> int:
        """
        Returns the node number at a specified position.

        :param position: position of the gene in the genome
        :return: node number where the gene belongs to
        """
        return (
            math.floor(position / (self.config.max_arity + 1)) + self.config.num_inputs
        )

    def node_position(self, node_num: int) -> int:
        """
        Returns the position of a node based on its number.

        :param node_num: number of the node in the genome
        :return: node position in the genome
        """
        return (node_num - self.config.num_inputs) * (self.config.max_arity + 1)

    def node_function(self, node_num: int, genome: list[int]) -> int:
        """
        Returns the function of the node at the given node number.

        :param node_num: Number of the node
        :param genome: Genome of the individual
        :return: function gene value
        """
        position = self.node_position(node_num)
        return genome[position]

    def node_connections(self, node_num: int, genome: list[int]) -> list:
        """
        Get the input connection genes of a node at the given node number.

        :param node_num: Number of the node
        :param genome: Genome of the individual
        :return: List connection gene values
        """
        position = self.node_position(node_num)
        return genome[position + 1 : position + self.config.max_arity + 1]

    def get_outputs(self, genome: list[int]) -> list[int]:
        """
        Return the output genes.

        :param genome: Genome of an individual
        :return: List of output genes
        """
        return genome[len(genome) - self.config.num_outputs : len(genome)]

    def max_gene(self, position: int):
        """
        Return the maximum gene value that is specified at a
        position in the genome.

        :param position: Gene position
        :return: Maximum gene value
        """
        if self.phenotype(position) == self.GeneType.OUTPUT:
            return self.config.num_inputs + self.config.num_functions - 1
        elif self.phenotype(position) == self.GeneType.CONNECTION:
            return self.config.num_functions - 1
        else:
            return self.node_number(position) - 1

    def evaluate_individual(self, genome: list[int], problem) -> float:
        """
        Evaluates an individual against the problem.

        :param genome: the genome of an individual
        :return: fitness of the individual
        """
        self.current_paths = None
        return problem.evaluate(genome, self)

    def evaluate_observation(self, genome: list[int], observation):
        """
        Evaluates an observation (of a dataset or environment)

        :param genome: Genome of an individual
        :param observation: Given observation
        :return: Prediction based on the genome and observation
        """
        paths = self.decode(genome)
        return self.predict(genome, observation, paths)

    def evaluate_node(self, node_num: int, genome: list[int], args: list) -> float:
        """
        Evaluates one node of the genome in the framework of the evaluation process.

        :param node_num: Number of the node
        :param genome: Genome of the individual that is evaluated
        :param args: Arguments given to the node

        :return: function value of the output of the node
        """
        function = self.node_function(node_num, genome)
        arity = self.functions[function].arity
        if arity < len(args):
            args = args[:arity]
        return self.functions[function](*args)

    def eval_complexity(self, genome: list[int]) -> float:
        """
        Returns the complexity of the genome based on the number of active nodes.

        :param genome: Genome of an individual
        :return: Complexity value
        """
        active_nodes = self.active_nodes(genome)
        return len(active_nodes)

    def is_valid(self, genome: list[int]) -> bool:
        """ """
        return True

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

        # if self.current_paths is None:
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

    def predict_iter(self, genome: list[int], observation: list) -> list:
        """
        Iterative forward prediction for CGP that calculates and stores the node values
        iteratively. The calculated node values are stored in a list and reused when the
        function node is evaluated again as an node argument.

        :param genome: The genome of the individual
        :param observation: Current observation
        :param paths: Decoded paths of the graph
        :return: Set of predicted values
        """
        max_idx = self.config.genes_per_node * self.config.num_function_nodes
        step = self.config.genes_per_node

        node_values = [
            None for i in range(self.config.num_function_nodes + self.config.num_inputs)
        ]
        for i in range(self.config.num_inputs):
            if not self.terminals[i].const:
                node_values[i] = observation[i]
            else:
                node_values[i] = self.terminals[i]()

        n_idx = self.config.num_inputs
        for node_num in range(0, max_idx, step):
            function = genome[node_num]
            args = [
                node_values[i]
                for i in genome[
                    node_num + 1 : node_num + self.function2arity[function] + 1
                ]
            ]
            node_values[n_idx] = self.functions[function](*args)
            n_idx += 1

        return [
            node_values[genome[max_idx + i]] for i in range(self.config.num_outputs)
        ]

    def active_nodes(self, genome: list[int], reverse=False) -> list[int]:
        """
        Determines the active nodes of an CGP individual and stores the node
        number.

        :param genome: Genome of the individual
        :param reverse: Reverse ordering of node numbers
        :return: Set of active node number
        """
        nodes_active = dict()

        # All function nodes referenced by the output genes
        # are active nodes
        for node_num in self.get_outputs(genome):
            if node_num >= self.config.num_inputs:
                nodes_active[node_num] = True

        start = self.config.num_genes - self.config.num_outputs - 1
        stop = 0
        step = -1
        # Iterate backwards over all genes of the function nodes
        for position in range(start, stop, step):
            node_num = self.node_number(position)
            # Continue only if the current position is a connection genes
            # and when the node is already known to be active
            if (
                node_num in nodes_active.keys()
                and self.phenotype(position) == self.GeneType.CONNECTION
            ):
                gene = genome[position]
                # Store only node numbers of active function nodes
                if gene >= self.config.num_inputs:
                    nodes_active[gene] = True
        return sorted(nodes_active.keys(), reverse=reverse)

    def decode(self, genome: list[int]) -> list[list[int]]:
        """
        Decodes the paths of the given genome and stores these as sequences
        of active function nodes.

        :param genome: Genome of an individual
        :return: decoded paths
        """
        paths = []
        node_map = dict()
        step = -self.config.genes_per_node

        # Iterate over the outputs of the genome
        for node_num in self.get_outputs(genome):
            node_map.clear()
            node_map[node_num] = True

            start = self.config.num_genes - self.config.num_outputs - 1
            stop = 0
            # Iterate backwards over the genes of the function nodes
            for gene_pos in range(start, stop, step):
                node_num = self.node_number(gene_pos)
                # Continue only if the node is linked to the path that
                # leads to the current output
                if node_num in node_map:
                    # Iterate over the connection genes
                    for connection in self.node_connections(node_num, genome):
                        # Store the node status of the connections in
                        # the node map
                        node_map[connection] = True
            path = sorted(node_map.keys())
            paths.append(path)
        return paths

    def decode_optimized(self, genome: list[int]) -> list[list[int]]:
        """
        Optimized decoding variant for problems with a high number of outputs
        that makes use of the active nodes determination to obtain the paths in the graph
        more efficiently.

        :param genome: Genome of the individuals
        :return: paths of the graph
        """
        nodes_active = self.active_nodes(genome, reverse=True)
        paths = []
        node_map = dict()

        # Iterate over all outputs
        for output in self.get_outputs(genome):
            node_map.clear()
            node_map[output] = True

            # Visit all active nodes that are linked with path that
            # lead to the current output
            for node_num in nodes_active:
                if node_num in node_map:
                    for connection in self.node_connections(node_num, genome):
                        # Use a node map to track the nodes
                        node_map[connection] = True
            # Obtain the path from the node map
            path = sorted(node_map.keys())
            paths.append(path)
        return paths

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

    def mutation(self, genome: list[int]):
        """
        Performs the standard point mutation that is commonly used
        with CGP. A selected gene is resampled uniformly in its valid range.

        :param genome: Genome of the individual to be mutated
        :return: the mutated genome
        """
        if self.hyperparameters.mutation_rate is not None:
            num_genes = int(self.hyperparameters.mutation_rate * self.config.num_genes)
            for _ in range(num_genes):
                gene_pos = random.randint(0, self.config.num_genes - 1)
                genome[gene_pos] = self.init_gene(gene_pos)
        else:
            raise ValueError("The mutation_rate must be set")

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


def print_population(self):
    for individual in self.population:
        self.print_individual(individual)


def print_individual(self, individual):
    print(f"Genome: {individual.genome} Fitness: {individual.fitness}")
