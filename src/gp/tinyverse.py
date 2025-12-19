"""
TinyverseGP: A modular cross-domain benchmark system for Genetic Programming.
             The tinyverse module includes a collections of base classes to be used by the GP modules.
             and therefore represent the fundamental model architecture of TinyverseGP.

             - GPModel: Abstract base class for GP representations
             - Config: Abstract base class for the GP configuration
                - GPConfig: Basic config class that considers fundamental
                            settings needed to run GP
             - Hyperparameters: Abstract base class for the hyperparameters
                - GPHyperparameters: Basic HP's often needed for the configuration of GP
             - Function: Base class for the representation of a non-terminal symbol
                         which is part of the function set
                    - Var:  Derived class for the representation of variable terminal symbols
                    - Const: Derived class for the representation of const terminal symbols
"""

import random
import time
import os
import json
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass, field, fields
from typing import List, Any, Generic, Callable, Dict
import copy

from src.gp.types import HPType
import yaml
import dill


class GPIndividual(ABC):
    genome: any
    fitness: any
    evaluated: bool

    """
    Class that is used to represent a GP individual.
    Formally a GP individual can be represented as a tuple consisting of
    the genome and the fitness value.
    """

    def __init__(self, genome_: any = None, fitness_: any = None):
        self.genome = genome_
        self.fitness = fitness_

        if fitness_ is None:
            self.evaluated = False

    def list(self):
        return [self.genome, self.fitness]

    @abstractmethod
    def serialize_genome(self):
        pass

    @abstractmethod
    def deserialize_genome(self):
        pass


@dataclass
class Config(ABC):
    """
    Abstract class for configuration classes.
    """

    def as_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict):
        names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass(kw_only=True)
class GPConfig(Config):
    """
    Configuration class for GP models.
    This class contains the common configuration parameters for GP models related to
    execution and output of a run.
    """

    global_seed: int
    num_jobs: int
    max_generations: int
    stopping_criteria: float
    minimizing_fitness: bool
    ideal_fitness: float
    silent_algorithm: bool
    silent_evolver: bool
    minimalistic_output: bool
    num_outputs: int
    report_interval: int
    max_time: int
    constraints: Callable[[Any], float] = lambda x: 0.0
    checkpointing: bool = False
    checkpoint_interval: int = 0
    checkpoint_dir: str = ""
    experiment_name: str = ""


@dataclass
class Hyperparameter(ABC, Generic[HPType]):
    name: str
    lower: HPType
    upper: HPType
    type: HPType


@dataclass
class HyperparameterSpace(ABC):
    space: list[Hyperparameter]


@dataclass(kw_only=True)
class Hyperparameters(ABC):
    """
    Base class for the GP hyperparamters.
    """

    penalization_complexity_factor: float = 0.0
    penalization_feasibility_factor: float = 0.0
    penalization_validity_factor: float = 0.0
    discard_invalid: bool = True
    discard_infeasible: bool = False

    def __post_init__(self):
        self.space = dict()

    def to_yaml(self):
        with open("hp.yml", "w") as file:
            yaml.dump(self.space, file, default_flow_style=False)

    def as_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict):
        names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass(kw_only=True)
class GPHyperparameters(Hyperparameters):
    """
    Hyperparameters class for Genetic Programming models.
    This class is responsible for storing the tunable hyperparameters of the GP model.
    """

    pop_size: int
    mutation_rate: float
    cx_rate: float
    tournament_size: int

    def __post_init__(self):
        Hyperparameters.__post_init__(self)
        self.space["pop_size"] = (10, 5000)
        self.space["mutation_rate"] = (0.0, 1.0)
        self.space["cx_rate"] = (0.0, 1.0)
        self.space["tournament_size"] = (2, 9)
        self.space["penalization_complexity_factor"] = (0.0, 1.0)
        self.space["penalization_feasibility_factor"] = (0.0, 1.0)
        self.space["penalization_validity_factor"] = (0.0, 1.0)
        self.space["discard_invalid"] = (False, True)
        self.space["discard_infeasible"] = (False, True)


@dataclass
class Function:
    """
    Main class used for representing functions, variables or constants.
    It contains information about arity, a string representation for the function,
    and the function itself.
    The method `call` is used to call the function with the given arguments.
    The method `custom` is meant to make the function compatible with sympy, if
    None is passed, it will use the defaults.
    """

    name: str
    arity: int
    function: callable
    custom: callable

    def __init__(self, arity_, name_, function_, custom_=None):
        self.function = function_
        self.name = name_
        self.arity = arity_
        self.custom = custom_

    def __call__(self, *args) -> Any:
        assert len(args) == self.arity
        return self.function(*args)


class Var(Function):
    """
    Class for representing variable terminals.
    """

    def __init__(self, index: int = None, name_: str = None):
        self.const = False
        if name_ is None:
            name_ = "Var"
        Function.__init__(self, 0, name_, lambda: index)


class Const(Function):
    """
    Class for representing constant terminals.
    """

    def __init__(self, value):
        self.const = True
        Function.__init__(self, 0, "Const", lambda: value)


@dataclass
class GPState:
    generation: int
    evaluations: int
    erc: list
    population: list


class Checkpointer:

    hyperparameters: Hyperparameters
    config: GPConfig

    def __init__(self, config_, hyperparameters_):
        self.config = config_
        self.hyperparameters = hyperparameters_
        self.path = self.create_dir()

    def write(self, state):

        def dump_population(population):
            l = []
            for ind in population:
                l.append([ind.serialize_genome(), ind.fitness])
            return l

        file_path = self.path + "/checkpoint_gen_" + str(state.generation) + ".dill"

        checkpoint = {
            "generation": state.generation,
            "evaluations": state.evaluations,
            "config": self.config.as_dict(),
            "hyperparameters": self.hyperparameters.as_dict(),
            "erc": state.erc,
            "population": dump_population(state.population),
        }

        outfile = os.open(file_path, flags=os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.write(outfile, dill.dumps(checkpoint))

    def load(self, file):
        with open(file, "rb") as infile:
            checkpoint = dill.load(infile)
        if not self.config.silent_evolver:
            print(f"Checkpoint {file} successfully loaded")
        return checkpoint

    def create_dir(self):
        dir_name = (
            self.config.experiment_name
            if self.config.experiment_name is not None
            else time.time()
        )
        path = os.path.join(self.config.checkpoint_dir, dir_name)

        if not os.path.exists(path):
            os.makedirs(path)

        return path


class GPModel(ABC):
    """
    Abstract based class for tiny GP representation modules.
    It describes the minimum requirements for a GP model that is
    integrated in the framework.
    """

    best_individual: GPIndividual
    num_evaluations: int
    generation_number: int
    population: List[GPIndividual]
    erc: list
    hyperparameters: GPHyperparameters
    config: GPConfig

    def __init__(self, config_: GPConfig, hyperparameters_: GPHyperparameters):
        # self.best_individual = GPIndividual()
        self.config = config_
        self.hyperparameters = hyperparameters_
        self.erc = []
        self.checkpointer = Checkpointer(self.config, self.hyperparameters)
        self.generation_number = 0
        self.num_evaluations = 0
        random.seed(self.config.global_seed)

    @abstractmethod
    def init_population(self):
        """
        Creates the population the way it is done for the respective GP representation.
        """
        pass

    def evaluate(self, problem) -> GPIndividual:
        """
        Evaluates the population.

        :returns: the best solution discovered in the population
        """
        best = None
        for individual in self.population:
            self.num_evaluations += 1  # update the evaluation counter
            genome = individual.genome
            if individual.fitness is None:
                # Dynamic dispatch: pass individual for Tiny3GE, genome for others
                if self.__class__.__name__ == 'Tiny3GE':
                    individual.fitness = self.penalize(
                        self.evaluate_individual(individual, problem), individual
                    )
                else:
                    individual.fitness = self.penalize(
                        self.evaluate_individual(genome, problem), genome
                    )
            fitness = individual.fitness

            if best is None:
                best = copy.copy(individual)
                best_fitness = fitness

            if problem.is_better(fitness, best_fitness):
                best = copy.copy(individual)
                best_fitness = fitness

        self.best_individual = copy.copy(best)

        return best

    def penalize(self, fitness: float, genome: GPIndividual) -> float:
        """
        Penalizes the fitness of a genome.
        This method is used to penalize genomes that are
        invalid, do not satisfy the constraints of the problem, or are too complex.
        """
        valid = self.is_valid(genome)
        if self.hyperparameters.discard_invalid and not valid:
            if self.config.minimizing_fitness:
                return float("inf")
            else:
                return float("-inf")
        violations = self.config.constraints(genome)
        if self.hyperparameters.discard_infeasible and violations > 0:
            if self.config.minimizing_fitness:
                return float("inf")
            else:
                return float("-inf")
        complexity = self.eval_complexity(genome)

        penalty = (
            self.hyperparameters.penalization_complexity_factor * complexity
            + self.hyperparameters.penalization_feasibility_factor * violations
            + self.hyperparameters.penalization_validity_factor * (1.0 - valid)
        )

        if self.config.minimizing_fitness:
            fitness += penalty
        else:
            fitness -= penalty
        return fitness

    @abstractmethod
    def is_valid(self, genome: GPIndividual) -> bool:
        """
        Checks if the genome is valid.
        """
        pass

    @abstractmethod
    def eval_complexity(self, genome: GPIndividual) -> float:
        """
        Evaluates the complexity of the genome.
        """
        pass

    @abstractmethod
    def evaluate_individual(self, genome: GPIndividual) -> float:
        """
        Fitness function that evaluates a single individual.
        """
        pass

    @abstractmethod
    def pipeline(self, problem):
        """
        Function that executes a single step of the evolutionary process.
        """
        pass


    def reset(self):
        """
        Resets the population and the counters.
        If multiple instances should be executed with different seeds
        the rng is re-seeded.
        """

        self.init_population()

        self.num_evaluations = 0
        self.generation_number = 0

        # Re-seed the random generator if multiple jobs are performed with
        # different seeds
        if self.config.num_jobs > 1 and self.config.global_seed is None:
            random.seed()


    def evolve(self, problem) -> Any:
        """
        Main evolution loop that is used to run instances
        of a GP model.

        The method can report the current state generation or/and
        job-wise. The best solution of the job is returned after all jobs have been
        completed.
        """
        best_individual = None
        best_fitness = None

        for job in range(self.config.num_jobs):

            # Reset the state
            self.reset()

            t0 = time.time()
            elapsed = 0
            terminate = False

            # Evaluate the population
            best_individual = self.evaluate(problem)
            best_fitness = best_fitness_job = best_individual.fitness

            while self.generation_number < self.config.max_generations:

                best_gen = self.pipeline(problem)
                best_gen_fitness = best_gen.fitness

                if problem.is_better(best_gen_fitness, best_fitness):
                    best_individual = best_gen
                    best_fitness = best_gen_fitness

                if problem.is_better(best_gen_fitness, best_fitness_job):
                    best_fitness_job = best_gen_fitness

                self.report_generation(
                    silent_algorithm=self.config.silent_algorithm,
                    generation=self.generation_number,
                    best_fitness=best_fitness,
                    report_interval=self.config.report_interval,
                )

                if self.config.checkpointing:
                    if (
                        self.generation_number > 0
                        and self.generation_number % self.config.checkpoint_interval == 0
                    ):
                        self.checkpointer.write(self.state())

                if problem.is_ideal(best_gen_fitness):
                    if not self.config.silent_algorithm:
                        print(
                            f"Ideal fitness found in generation {self.generation_number}"
                        )
                    break

                if (
                    self.generation_number & 15
                ) == 0:  # check periodically if the time limit is reached
                    t1 = time.time()
                    delta = t1 - t0
                    t0 = t1
                    elapsed += delta
                    if elapsed + delta >= self.config.max_time:
                        if not self.config.silent_algorithm:
                            print("Timelimit exceeded")
                        break

                self.generation_number += 1

            self.report_job(
                job=job,
                num_evaluations=self.num_evaluations,
                best_fitness=best_fitness_job,
                silent_evolver=self.config.silent_evolver,
                minimalistic_output=self.config.minimalistic_output,
            )

            if terminate:
                break

        return best_individual
    
    @abstractmethod
    def selection(self) -> Any:
        """
        Implementation of the selection mechanism.
        Commonly returns an individual object or the position
        of an individual in the population.
        """
        pass

    @abstractmethod
    def predict(self) -> Any:
        """
        The respective prediction method is implemented here.
        """
        pass

    @abstractmethod
    def expression(self) -> Any:
        """
        Returns a human-readable solution of a evolved candidate solution.
        Return value can be a string or a list of strings.
        """
        pass

    def state(self) -> GPState:
        return GPState(
            erc=self.erc,
            population=self.population,
            generation=self.generation_number,
            evaluations=self.num_evaluations,
        )

    def resume(self, checkpoint, problem):
        self.generation_number = checkpoint["generation"]
        self.num_evaluations = checkpoint["evaluations"]
        population = checkpoint["population"]

        for idx, ind in enumerate(population):
            self.population[idx].deserialize_genome(ind[0])
            self.population[idx].fitness = ind[1]

        if not self.config.silent_evolver:
            print(f"Resuming from generation {self.generation_number}")
        self.evolve(problem)

    def report_job(
        self,
        job: int,
        num_evaluations: int,
        best_fitness: float,
        silent_evolver: bool,
        minimalistic_output: bool,
    ):
        """
        Report the status of a job after has been executed.

        :param job: job number
        :param num_evaluations: Number of evaluations the run lasted
        :param best_fitness: Best fitness found during the run
        :param silent_evolver: Switch for activating/deactivating the report function
        :param minimalistic_output: Swith for minimalistic output
        :return:
        """
        if not silent_evolver:
            if not minimalistic_output:
                print(
                    "Job #"
                    + str(job)
                    + " - Evaluations: "
                    + str(num_evaluations)
                    + " - Best Fitness: "
                    + str(best_fitness)
                )
            else:
                print(str(num_evaluations) + ";" + str(best_fitness))

    def report_generation(
        self,
        silent_algorithm: bool,
        generation: int,
        best_fitness: float,
        report_interval: int,
    ):
        """
        Reports the status of a generation.

        :param silent_algorithm: Switch for activating/deactivating the report function
        :param generation: Generation number
        :param best_fitness: Best fitness found so far
        :param report_interval: Interval after which the generation status is reported
        """
        if not silent_algorithm and generation % report_interval == 0:
            print(
                "Generation #"
                + str(generation)
                + " - Best Fitness: "
                + str(best_fitness)
            )
