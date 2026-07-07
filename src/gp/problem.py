"""
Implements various problem types supported by TinyverseGP to
approach various problem domains with the benchmarking package.

Currently, the following problem types are provided:

- BlackBox: Used for symbolic regression and logic synthesis
- PolicySearch: Used for reinforcement learning problems
- ProgramSynthesis: Used for the provided coding problems

"""
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from abc import ABC
from collections.abc import Callable
from src.benchmark.policy_search.policy_evaluation import GPAgent
from src.gp.tinyverse import GPModel
import numbers
import ioh
import atexit

class Problem(ABC):
    """
    Abstract base class for the problems supported by TinverseGP.
    Provides a skeleton for the child classes that provide actual
    implementations of problems.
    """
    __protected_attributes__ = (
        "logger",
        "num_evaluations",
        "reset",
        "meta_data",
        "log_evaluation"
    )
    ideal: float
    logger: ioh.logger.AbstractLogger

    def __init__(self, fid, name, minimizing: bool = True, logger: ioh.logger.AbstractLogger = None):
        self.logger = logger
        self.minimizing = minimizing
        self.num_evaluations = 0
        if logger is None:
            return
        self.meta_data = ioh.MetaData(
            fid,
            1,
            name,
            1,
            ioh.OptimizationType.MIN if minimizing else ioh.OptimizationType.MAX,
        )
        self.logger.attach_problem(self.meta_data)
        atexit.register(self.logger.close)


    def log_evaluation(self, f1, x):
            self.num_evaluations += 1
            if self.logger is None:
                return
            info = ioh.LogInfo(
                self.num_evaluations,
                f1,
                0,
                0,
                0,
                0,
                0,
                [0], #If we want to log the genome, we need to log an ID here, and store the {ID, genome}-pairs in a separate file (or think of a better solution)
                [],
                [],
                ioh.iohcpp.RealSolution([], 0),
                True,
            )
            self.logger.call(info)

    def reset(self):
        self.num_evaluations = 0
        if self.logger is None:
            return
        self.logger.reset()
        self.logger.attach_problem(self.meta_data)

    def is_ideal(self, fitness: float) -> bool:
        """
        Check if the fitness reached an ideal state.
        This can prompt an early stop in the optimization process.

        :param fitness: fitness of the candidate individual
        :return: ideal fitness status
        """
        return fitness <= self.ideal if self.minimizing else fitness >= self.ideal

    def is_better(self, fitness1: float, fitness2: float) -> bool:
        """
        Check if the first fitness is better than the second.
        It takes into consideration whether the problem is minimizing or maximizing.

        :param fitness1: fitness of the first individual
        :param fitness2: fitness of the second individual
        :return: status if fitness1 is better than fitness2
        """
        return fitness1 < fitness2 if self.minimizing else fitness1 > fitness2

    def evaluate(self, genome, model: GPModel):
        """
        This method implements how to evaluate the genome using the model.
        It is problem-specific and should be implemented by the user.

        :param genome: Genome of the individual
        :param model: The respective GP model that is used
        """
        pass
    
    def is_stop(self):
        """
        Check for additional stop conditions besides the optimum of the defined problem.
        By default, there are no additional stop conditions in place.
        """
        return False
    
    def __del__(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.close()

@dataclass
class BlackBox(Problem):
    """
    Represents a black-box problem where the fitness is calculated by a loss function
    on a dataset of observations and the actual (objective) function values.
    """

    observations: list
    actual: list

    def __init__(
            self,
            observations_: list,
            actual_: list,
            loss_: Callable[[list, list], float],
            ideal_: float,
            minimizing_: bool,
            fid: int = 1,
            name: str = "None",
            logger: ioh.logger.AbstractLogger = None
    ):
        super().__init__(fid, name, minimizing_, logger=logger)
        self.observations = observations_
        self.actual = actual_
        self.loss = loss_
        self.ideal = ideal_
        self.minimizing = minimizing_
        self.unidim = True if isinstance(self.actual[0], numbers.Number) else False

    def evaluate(self, genome, model: GPModel) -> float:
        """
        Evaluates the genome of a GP individual on the dataset with the
        given GP model.

        The predictions made by the model are used to calculate the cost
        function value.

        :param genome: Genome of the individual
        :param model: Selected GP model

        :return: cost function value
        """
        predictions = []
        for observation in self.observations:
            prediction = model.predict(genome, observation)
            predictions.append(prediction)
        result = self.cost(predictions)
        self.log_evaluation(result, genome)
        return result
    
    def cost(self, predictions: list) -> float:
        """
        Calculates the cost function value based on the
        selected loss function that has been passed to the class.

        :param predictions: Set of predictions made by the model
        :return: cost function value
        """
        cost = 0.0

        for index, _ in enumerate(predictions[0]):
            cost += self.loss(
                [prediction[index] for prediction in predictions],
                [act if self.unidim else act[index] for act in self.actual],
            )
        return cost


class PolicySearch(Problem):
    """
    Representation of the policy search problem where the fitness is calculated by the
    average reward of the policy.

    A instance of the GPAgent class is used to evaluate a candidate policy within the respective
    environment.
    """

    agent: GPAgent
    num_episodes: int
    max_steps: int
    step_cnt: int
    frames_per_step = int

    def __init__(
            self, env: gym.Env, ideal_: float, minimizing_: bool, num_episodes_: int = 100,
            frames_per_step_: int = 1,
            max_steps_: float = 2e6, fid: int = 1, name: str = "None", logger: ioh.logger.AbstractLogger = None):
        super().__init__(fid=fid, name=name, minimizing=minimizing_, logger=logger)
        self.agent = GPAgent(env)
        self.ideal = ideal_
        self.minimizing = minimizing_
        self.num_episodes = num_episodes_
        self.frames_per_step = frames_per_step_
        self.max_steps = int(max_steps_)
        self.step_cnt = 0

    def is_stop(self):
        return self.step_cnt >= self.max_steps

    def evaluate(
            self, genome, model: GPModel, num_episodes: int = None, wait_key: bool = False
    ) -> float:
        if num_episodes is None:
            num_episodes = self.num_episodes
        reward, num_steps = self.agent.evaluate_policy(genome, model, num_episodes, wait_key)

        if self.frames_per_step is not None:
            self.step_cnt += num_steps * self.frames_per_step
        else:
            self.step_cnt += num_steps
        self.log_evaluation(reward, genome)
        return reward


class ProgramSynthesis(Problem):
    """
    Represent a program synthesis problem where the evaluation is based on a dataset
    that consists of positive examples and counterexamples.

    The predictions are made with a binary step activation function and the cost function
    is based on the hamming distance between the binary predictions and the actual values in the
    dataset.
    """

    def __init__(self, dataset_, minimizing_: bool = False,
                 fid: int = 1, name: str = "None", logger: ioh.logger.AbstractLogger = None):
        super().__init__(fid=fid, name=name, minimizing=minimizing_, logger=logger)
        self.dataset = dataset_
        self.minimizing = minimizing_

    def is_ideal(self, fitness):
        return fitness == len(self.dataset)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def binary_step(self, x, threshold=0.5):
        return 0 if self.sigmoid(x) <= threshold else 1

    def evaluate(self, genome, model):
        predictions = []
        for observation in self.dataset:
            prediction = model.predict(genome, observation)
            prediction = self.binary_step(prediction[0])
            predictions.append(prediction)
        result = self.cost(predictions)
        self.log_evaluation(result, genome)
        return result

    def cost(self, predictions):
        cost = 0
        for i, prediction in enumerate(predictions):
            cost += 1 if prediction == self.dataset[i][1] else 0
        return cost
