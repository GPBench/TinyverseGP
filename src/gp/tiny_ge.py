"""
TinyGP: A minimalistic implementation of Grammatical Evolution GE for
         TinyverseGP.
"""

import random
import copy
import time
import re
from src.gp.problem import *
from src.gp.tinyverse import *


@dataclass
class GEHyperparameters(GPHyperparameters):
    """
    Specialized hyperparameter configuration space for GE.
    """
    genome_length: int
    codon_size: int
    penalty_value: float

    def __post_init__(self):
        GPHyperparameters.__post_init__(self)
        self.space["genome_length"] = (10, 1000)
        self.space["codon_size"] = (1000, 25000)
        self.space["pop_size"] = (10, 5000)
        self.space["mutation_rate"] = (0.0, 1.0)
        self.space["cx_rate"] = (0.0, 1.0)
        self.space["tournament_size"] = (2, 9)
        self.space["penalty_value"] = (1, 9999999)

class GEIndividual(GPIndividual):
    genome: list[int]
    fitness: any

    def __init__(self, genome_: list[int], fitness_: any = None):
        GPIndividual.__init__(self, genome_, fitness_)


    def serialize_genome(self):
        return self.genome

    def deserialize_genome(self, genome_):
        self.genome = genome_

class TinyGE(GPModel):
    """
    Main class of the tiny GE module that derives from GPModel and
    implements all related fundamental mechanisms to run GE.
    """

    config: Config
    hyperparameters: Hyperparameters
    problem: Problem
    functions: list[Function]


    def __init__(self, functions_: list[Function], grammar_: dict, arguments_: list[str], config: Config, hyperparameters: Hyperparameters):
        super().__init__(config, hyperparameters)
        self.functions = {f.name.upper(): f.function for f in functions_} # the list of functions to that could be used in the grammar                                 # TODO: Adjust to updates in the framework
        self.grammar = grammar_ # the defined grammar
        self.arguments = arguments_ # the arguments for the functions to be generated
        self.hyperparameters = hyperparameters # hyperparameters
        self.config = config # overall configuration
        self.best_individual = None  # to keep the best program found so far
        self.num_evaluations = 0  # counter of number of evaluations
        self.init_population()

    def init_population(self):
        # initial population using uniform initialization
        self.population = [GEIndividual(genome, None) for genome in
                           self.init_uniform(self.hyperparameters.pop_size, self.hyperparameters.genome_length,
                                             self.hyperparameters.codon_size)]

    def init_uniform(self, num_pop: int, max_genome_length: int, codon_size: int):
        """
        Initialize the population uniformly. It will create one genome per output.

        :return: a list of lists
        """
        pop = []
        for _ in range(num_pop):
            pop.append(
                [random.randint(0, codon_size) for _ in range(max_genome_length)]
            )
        return pop

    def evaluate_individual(self, genome:list[int], problem) -> float:
        '''
        Evaluate a single individual `genome`.

        :return: a `float` representing the fitness of that individual.
        '''
        f = None
        tmp_expr = self.expression(genome)
        if "<" in tmp_expr or ">" in tmp_expr:
            f = self.hyperparameters.penalty_value
        else:
            f = problem.evaluate(genome, self) # evaluate the solution using the problem instance
        if self.best_individual is None or problem.is_better(f, self.best_individual.fitness):
            self.best_individual = GEIndividual(genome, f)
        return f

    def eval_complexity(self, genome: list[int]) -> int:
        '''
        Returns the complexity of the genome.

        :return: an integer representing the number of nodes in the genome.
        '''
        count = 0
        tmp_genome = copy.deepcopy(genome)
        expression = "<expr>"
        while '<' in expression and len(tmp_genome) > 0:
            next_non_terminal = re.search(r'<(.*?)>', expression).group(0)
            choice = self.grammar[next_non_terminal][(tmp_genome.pop(0) % len(self.grammar[next_non_terminal]))]
            expression = expression.replace(next_non_terminal, choice, 1)
            count += 1
        return count

        return sum([node_size(g) for g in genome])

    def is_valid(self, genome: list[int]) -> bool:
        '''
        Check if the genome is valid. A genome is valid if it has the same number of outputs as the problem.

        :return: a boolean indicating whether the genome is valid or not.
        '''
        tmp_expr = self.expression(genome)
        return '<' not in tmp_expr and '>' not in tmp_expr

    def predict(self, genome: list, observation: list) -> list:
        """
        Predict the output of the `genome` given a single `observation`.

        :return: a list of the outputs for that observation
        """

        def evaluate_expression(
            expr: str, func_dict: list, args: list[str], values: list
        ) -> any:
            local_vars = dict(zip(args, values))
            prediction = eval(expr, func_dict, local_vars)
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

    def breed(self):
        """
        Breed the population by first selecting a set of pair of parents and then
        applying crossover and mutation operators.
        """
        # Select n pairs of parents using tournament selection, n is the population size minus 1 (so we have space for the best individual)
        parents = [
            [self.selection(), self.selection()]
            for _ in range(self.hyperparameters.pop_size - 1)
        ]
        # replace the current population by perturbing the sampled parents
        self.population = [self.perturb(*parent) for parent in parents]
        # keep the best solution in the population
        self.population.append(
            GEIndividual(self.best_individual.genome, self.best_individual.fitness)
        )

    def perturb(self, parent1: list, parent2: list) -> list:
        """
        Applies the crossover and mutation operators to the parents.

        :return: a list of the `genome` and `None` representing the unevaluated fitness.
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

    def selection(self) -> list:
        """
        Select a parent from the population using the tournament selection method.

        :return: a selected genome.
        """
        # samples `self.hyperparameters.tournament_size` solutions completely at random

        parents = [random.choice(self.population) for _ in range(self.hyperparameters.tournament_size)]
        # return the best of this sample whether it is a minimization or maximization problem     
        if self.config.minimizing_fitness:
            return min(parents, key=lambda ind: ind.fitness).genome
        else:
            return max(parents, key=lambda ind: ind.fitness).genome

    def crossover(self, p1: list, p2: list) -> list:
        """
        Applies onepoint crossover to the given parents.

        :return: the recombined genomes.
        """
        parent1 = copy.deepcopy(p1)
        parent2 = copy.deepcopy(p2)

        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return random.choice([child1, child2])

    def mutation(
        self, genome: list, min_val: int, max_val: int, mutation_rate: float
    ) -> list:
        """
        Apply the int flip per codon mutation to the parent.

        :return: the mutated genome.
        """
        mutated = copy.deepcopy(genome)

        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                old_val = mutated[i]
                possible_values = list(range(min_val, max_val + 1))
                possible_values.remove(old_val)
                mutated[i] = (
                    random.choice(possible_values) if possible_values else old_val
                )

        return mutated

    def genotype_phenotype_mapping(self, grammar, genome, expression="<expr>"):
        """
        Maps the genotype to its phenotype.

        :return: a string representation of the genome.
        """
        tmp_genome = copy.deepcopy(genome)
        while "<" in expression and len(tmp_genome) > 0:
            next_non_terminal = re.search(r"<(.*?)>", expression).group(0)
            choice = grammar[next_non_terminal][
                (tmp_genome.pop(0) % len(grammar[next_non_terminal]))
            ]
            expression = expression.replace(next_non_terminal, choice, 1)
        return expression

    def expression(self, genome: list) -> str:
        """
        Convert a genome into string format with the help of the grammar.

        :return: expression as `str`.
        """
        return self.genotype_phenotype_mapping(self.grammar, genome, "<expr>")

    def print_population(self):
        """
        Prints the entire population
        """
        for individual in self.population:
            self.print_individual(individual)

    def print_individual(self, individual):
        """
        Prints information about a single individual.
        """
        print(
            "Expression: "
            + ";".join(self.expression(individual[0]))
            + " : Fitness: "
            + str(individual[1])
        )

    def pipeline(self, problem):
        """
        Single step of TGP
        """

        self.breed()
        return self.evaluate(problem)
