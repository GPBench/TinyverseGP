"""
TinyGE: A minimalistic implementation of Grammatical Evolution (GE) for
        TinyverseGP.

Grammatical Evolution represents candidate programs as variable-length integer
genomes ("codon" lists). A genome is translated into an executable expression by
repeatedly mapping the left-most non-terminal of a context-free grammar onto one
of its productions; the concrete production is selected via ``codon % number_of_choices``.
This module provides the genome representation (:class:`GEIndividual`) together
with the full evolutionary machinery (:class:`TinyGE`) required to run GE inside
the framework.
"""

import random
import copy
import re
from dataclasses import dataclass
from typing import Any, Callable

from src.gp.problem import Problem
from src.gp.tiny_tgp import node_size
from src.gp.tinyverse import GPIndividual, GPHyperparameters, Config, Function, Hyperparameters, GPModel


@dataclass
class GEHyperparameters(GPHyperparameters):
    """
    Specialized hyperparameter configuration space for GE.

    Extends :class:`GPHyperparameters` with the settings that are specific to the
    grammatical-evolution genome representation.

    :ivar genome_length: Number of codons contained in a single genome.
    :ivar codon_size: Exclusive upper bound for the integer value of a codon.
    :ivar penalty_value: Fitness value assigned to genomes that cannot be fully
        mapped to a valid (terminal-only) expression.
    """

    genome_length: int
    codon_size: int
    penalty_value: float

    def __post_init__(self) -> None:
        """
        Initialize the tunable hyperparameter search space after dataclass creation.

        Registers the valid ranges (as ``(lower, upper)`` tuples) for every
        GE-specific and inherited hyperparameter so they can be sampled/tuned.

        :return: ``None``.
        """
        GPHyperparameters.__post_init__(self)
        # (lower, upper) bounds used when sampling/tuning each hyperparameter
        self.space["genome_length"] = (10, 1000)
        self.space["codon_size"] = (1000, 25000)
        self.space["pop_size"] = (10, 5000)
        self.space["mutation_rate"] = (0.0, 1.0)
        self.space["cx_rate"] = (0.0, 1.0)
        self.space["tournament_size"] = (2, 9)
        self.space["penalty_value"] = (1, 9999999)


class GEIndividual(GPIndividual):
    """
    Genome representation of a single grammatical-evolution individual.

    A GE individual is fully described by its integer codon genome and its
    fitness value.

    :ivar genome: The list of integer codons encoding the individual.
    :ivar fitness: The fitness value of the individual, or ``None`` if unevaluated.
    """

    genome: list[int]
    fitness: Any

    def __init__(self, genome_: list[int], fitness_: Any = None) -> None:
        """
        Create a new GE individual.

        :param genome_: List of integer codons that encode the individual.
        :param fitness_: Optional pre-computed fitness value; ``None`` marks the
            individual as unevaluated.
        :return: ``None``.
        """
        GPIndividual.__init__(self, genome_, fitness_)

    def serialize_genome(self) -> list[int]:
        """
        Serialize the genome for checkpointing.

        The codon list is already a plain, serializable structure, so it is
        returned unchanged.

        :return: The integer codon genome.
        """
        return self.genome

    def deserialize_genome(self, genome_: list[int]) -> None:
        """
        Restore the genome from a serialized representation.

        :param genome_: Previously serialized integer codon genome.
        :return: ``None``.
        """
        self.genome = genome_


class TinyGE(GPModel):
    """
    Main class of the tiny GE module that derives from :class:`GPModel` and
    implements all related fundamental mechanisms to run GE.

    :ivar config: Overall run configuration.
    :ivar hyperparameters: GE-specific hyperparameter configuration.
    :ivar problem: The problem instance the model is optimized against.
    :ivar functions: Mapping from upper-cased function name to the callable used
        while evaluating an expression.
    :ivar grammar: Context-free grammar mapping each non-terminal to its list of
        possible productions.
    :ivar arguments: Names of the input arguments available to generated programs.
    :ivar best_individual: Best individual discovered so far.
    :ivar num_evaluations: Number of fitness evaluations.
    :ivar population: Current population of :class:`GEIndividual` objects.
    """

    config: Config
    hyperparameters: Hyperparameters
    problem: Problem
    functions: dict[str, Callable]
    grammar: dict[str, list[str]]
    arguments: list[str]
    best_individual: GEIndividual
    num_evaluations: int
    population: list[GEIndividual]

    def __init__(
        self,
        functions_: list[Function],
        grammar_: dict[str, list[str]],
        arguments_: list[str],
        config: Config,
        hyperparameters: Hyperparameters,
    ) -> None:
        """
        Initialize the GE model and build its initial population.

        :param functions_: Function set whose entries may appear in the grammar.
        :param grammar_: Context-free grammar mapping non-terminals to productions.
        :param arguments_: Names of the input arguments available to generated programs.
        :param config: Overall run configuration.
        :param hyperparameters: GE-specific hyperparameter configuration.
        :return: ``None``.
        """
        super().__init__(config, hyperparameters)
        # Build a name -> callable lookup so expressions can reference functions by name
        self.functions = {f.name.upper(): f.function for f in functions_}  # the list of functions that could be used in the grammar  # TODO: Adjust to updates in the framework
        self.grammar = grammar_  # the defined grammar
        self.arguments = arguments_  # the arguments for the functions to be generated
        self.hyperparameters = hyperparameters  # hyperparameters
        self.config = config  # overall configuration
        self.best_individual = None  # to keep the best program found so far
        self.num_evaluations = 0  # counter of number of evaluations
        self.init_population()

    def init_population(self) -> None:
        """
        Create the initial population using uniform random initialization.

        :return: ``None``.
        """
        # initial population using uniform initialization
        self.population = [
            GEIndividual(genome, None)
            for genome in self.init_uniform(
                self.hyperparameters.pop_size,
                self.hyperparameters.genome_length,
                self.hyperparameters.codon_size,
            )
        ]

    def init_uniform(
        self, num_pop: int, max_genome_length: int, codon_size: int
    ) -> list[list[int]]:
        """
        Initialize the population uniformly. It will create one genome per output.

        :param num_pop: Number of genomes to generate.
        :param max_genome_length: Number of codons per genome.
        :param codon_size: Inclusive upper bound for each randomly drawn codon value.
        :return: A list of genomes, where each genome is a list of integer codons.
        """
        pop = []
        for _ in range(num_pop):
            # each codon is drawn uniformly from [0, codon_size]
            pop.append(
                [random.randint(0, codon_size) for _ in range(max_genome_length)]
            )
        return pop

    def evaluate_individual(self, genome: list[int], problem: Problem) -> float:
        """
        Evaluate a single individual ``genome``.

        Genomes that cannot be fully mapped to a terminal-only expression (i.e.
        that still contain grammar symbols) receive the configured penalty value
        instead of being executed.

        :param genome: Integer codon genome to evaluate.
        :param problem: Problem instance used to compute the fitness.
        :return: A ``float`` representing the fitness of that individual.
        """
        f = None
        tmp_expr = self.expression(genome)
        # An incompletely mapped genome still contains grammar brackets -> penalize
        if "<" in tmp_expr or ">" in tmp_expr:
            f = self.hyperparameters.penalty_value
        else:
            f = problem.evaluate(genome, self)  # evaluate the solution using the problem instance
        # Track the best individual seen so far across all evaluations
        if self.best_individual is None or problem.is_better(f, self.best_individual.fitness):
            self.best_individual = GEIndividual(genome, f)
        return f

    def eval_complexity(self, genome: list[int]) -> int:
        """
        Return the complexity of the genome.

        Complexity is measured as the number of production rules applied while
        mapping the genome onto its expression.

        :param genome: Integer codon genome whose complexity is measured.
        :return: An integer representing the number of nodes in the genome.
        """
        count = 0
        tmp_genome = copy.deepcopy(genome)  # work on a copy since codons are consumed
        expression = "<expr>"
        # Expand the left-most non-terminal until the expression is terminal or codons run out
        while '<' in expression and len(tmp_genome) > 0:
            next_non_terminal = re.search(r'<(.*?)>', expression).group(0)
            # Select the production via codon modulo the number of available choices
            choice = self.grammar[next_non_terminal][(tmp_genome.pop(0) % len(self.grammar[next_non_terminal]))]
            expression = expression.replace(next_non_terminal, choice, 1)
            count += 1
        return count

        return sum([node_size(g) for g in genome])

    def is_valid(self, genome: list[int]) -> bool:
        """
        Check if the genome is valid. A genome is valid if it maps to a complete,
        terminal-only expression (i.e. it contains no unexpanded grammar symbols).

        :param genome: Integer codon genome to validate.
        :return: A boolean indicating whether the genome is valid or not.
        """
        tmp_expr = self.expression(genome)
        return '<' not in tmp_expr and '>' not in tmp_expr

    def predict(self, genome: list[int], observation: list) -> list:
        """
        Predict the output of the ``genome`` given a single ``observation``.

        :param genome: Integer codon genome to evaluate.
        :param observation: The concrete input values bound to the program arguments.
        :return: A list of the outputs for that observation.
        """

        def evaluate_expression(
            expr: str, func_dict: dict, args: list[str], values: list
        ) -> list:
            """
            Evaluate a mapped expression string against a single observation.

            :param expr: The terminal expression produced by the grammar mapping.
            :param func_dict: Name -> callable lookup exposed to ``eval``.
            :param args: Names of the program arguments.
            :param values: Concrete values bound to ``args`` for this observation.
            :return: The prediction, always wrapped in a list.
            """
            # Bind argument names to their observed values for this evaluation
            local_vars = dict(zip(args, values))
            prediction = eval(expr, func_dict, local_vars)
            # Normalize scalar predictions to the list interface expected by callers
            if isinstance(prediction, list):
                return prediction
            else:
                return [prediction]

        tmp_expr = self.expression(
            genome
        )  # TODO: expression already generated in evaluate_individual() -> prevent double execution
        return evaluate_expression(
            tmp_expr, self.functions, self.arguments, observation
        )

    def breed(self) -> None:
        """
        Breed the population by first selecting a set of pairs of parents and then
        applying crossover and mutation operators.

        Elitism is applied: the best individual found so far is preserved into the
        next generation.

        :return: ``None``.
        """
        # Select n pairs of parents using tournament selection, n is the population size minus 1 (so we have space for the best individual)
        parents = [
            [self.selection(), self.selection()]
            for _ in range(self.hyperparameters.pop_size - 1)
        ]
        # replace the current population by perturbing the sampled parents
        self.population = [self.perturb(*parent) for parent in parents]
        # keep the best solution in the population (elitism)
        self.population.append(
            GEIndividual(self.best_individual.genome, self.best_individual.fitness)
        )

    def perturb(self, parent1: list[int], parent2: list[int]) -> GEIndividual:
        """
        Apply the crossover and mutation operators to the parents.

        :param parent1: First parent genome.
        :param parent2: Second parent genome.
        :return: An unevaluated :class:`GEIndividual` (fitness is ``None``).
        """
        # applies the crossover with `self.hyperparameters.cx_rate` probability, otherwise return the first parent
        genome = (
            self.crossover(parent1, parent2)
            if random.random() <= self.hyperparameters.cx_rate
            else parent1
        )
        # applies mutation with `self.hyperparameters.mutation_rate`
        genome = self.mutation(
            genome,
            0,
            self.hyperparameters.codon_size,
            self.hyperparameters.mutation_rate,
        )
        return GEIndividual(genome, None)  # returns the unevaluated offspring

    def selection(self) -> list[int]:
        """
        Select a parent from the population using the tournament selection method.

        :return: The genome of the selected individual.
        """
        # samples `self.hyperparameters.tournament_size` solutions completely at random
        parents = [random.choice(self.population) for _ in range(self.hyperparameters.tournament_size)]
        # return the best of this sample whether it is a minimization or maximization problem
        if self.config.minimizing_fitness:
            return min(parents, key=lambda ind: ind.fitness).genome
        else:
            return max(parents, key=lambda ind: ind.fitness).genome

    def crossover(self, p1: list[int], p2: list[int]) -> list[int]:
        """
        Apply one-point crossover to the given parents.

        :param p1: First parent genome.
        :param p2: Second parent genome.
        :return: One of the two recombined genomes, chosen at random.
        """
        parent1 = copy.deepcopy(p1)
        parent2 = copy.deepcopy(p2)

        # Split both parents at the same random point and swap the tails
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return random.choice([child1, child2])

    def mutation(
        self, genome: list[int], min_val: int, max_val: int, mutation_rate: float
    ) -> list[int]:
        """
        Apply the int-flip-per-codon mutation to the parent.

        Each codon is, with probability ``mutation_rate``, replaced by a different
        value drawn uniformly from ``[min_val, max_val]``.

        :param genome: Genome to mutate.
        :param min_val: Inclusive lower bound for a codon value.
        :param max_val: Inclusive upper bound for a codon value.
        :param mutation_rate: Per-codon probability of being mutated.
        :return: The mutated genome.
        """
        mutated = copy.deepcopy(genome)

        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                old_val = mutated[i]
                # Draw the replacement from all values except the current one
                possible_values = list(range(min_val, max_val + 1))
                possible_values.remove(old_val)
                mutated[i] = (
                    random.choice(possible_values) if possible_values else old_val
                )

        return mutated

    def genotype_phenotype_mapping(
        self,
        grammar: dict[str, list[str]],
        genome: list[int],
        expression: str = "<expr>",
    ) -> str:
        """
        Map the genotype to its phenotype.

        Starting from ``expression``, the left-most non-terminal is repeatedly
        replaced by the production selected through ``codon % number_of_choices``
        until the expression is terminal or the genome is exhausted.

        :param grammar: Context-free grammar mapping non-terminals to productions.
        :param genome: Integer codon genome that drives the production choices.
        :param expression: Start symbol / partial expression to expand.
        :return: A string representation of the genome.
        """
        tmp_genome = copy.deepcopy(genome)  # copy since codons are consumed during mapping
        while "<" in expression and len(tmp_genome) > 0:
            next_non_terminal = re.search(r"<(.*?)>", expression).group(0)
            # Select the production for the current non-terminal via codon modulo
            choice = grammar[next_non_terminal][
                (tmp_genome.pop(0) % len(grammar[next_non_terminal]))
            ]
            expression = expression.replace(next_non_terminal, choice, 1)
        return expression

    def expression(self, genome: list[int]) -> str:
        """
        Convert a genome into string format with the help of the grammar.

        :param genome: Integer codon genome to translate.
        :return: The mapped expression as ``str``.
        """
        return self.genotype_phenotype_mapping(self.grammar, genome, "<expr>")

    def print_population(self) -> None:
        """
        Print the entire population.

        :return: ``None``.
        """
        for individual in self.population:
            self.print_individual(individual)

    def print_individual(self, individual: GEIndividual) -> None:
        """
        Print information about a single individual (its expression and fitness).

        :param individual: The individual to print.
        :return: ``None``.
        """
        print(
            "Expression: "
            + ";".join(self.expression(individual[0]))
            + " : Fitness: "
            + str(individual[1])
        )

    def pipeline(self, problem: Problem) -> GPIndividual:
        """
        Perform a single step (generation) of GE.

        A generation consists of breeding a new population and evaluating it.

        :param problem: Problem instance used to evaluate the new population.
        :return: The best individual of the freshly evaluated population.
        """
        self.breed()
        return self.evaluate(problem)
