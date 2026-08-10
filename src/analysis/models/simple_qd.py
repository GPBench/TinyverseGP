import operator
import random
from abc import ABC, abstractmethod
from typing import Callable
from src.gp.problem import Problem
from src.gp.tinyverse import GPIndividual, GPModel


class SimpleQD(GPModel):
    m: dict
    y: GPIndividual
    cx_rate: float
    minimizing_fitness: bool

    def __init__(self, functions_, terminals_, config_, hyperparameters_):
        super().__init__(functions_, terminals_, config_, hyperparameters_)
        self.m = {}
        self.cx_rate = self.hyperparameters.cx_rate
        self.minimizing_fitness = self.config.minimizing_fitness
        self.best_individual = None

    @abstractmethod
    def behavior(self, y: GPIndividual):
        pass

    @abstractmethod
    def genome(self, x: GPIndividual):
        pass

    @abstractmethod
    def clone(self, x: GPIndividual):
        pass

    def selection(self) -> GPIndividual:
        return random.choice(list(self.m.values()))

    def best(self) -> GPIndividual:
        return sorted(list(self.m.values()), key=lambda x: x.fitness,
                      reverse=not self.minimizing_fitness)[0]

    def is_better(self, y: GPIndividual, x: GPIndividual):
        if self.minimizing_fitness:
            comp = operator.ge
        else:
            comp = operator.le

        if comp(y.fitness, x.fitness):
            return False
        return True

    def update(self, y: GPIndividual):
        """
        Updates the map with the fitness-behavior tuple obtained from a GPIndividual.
        The update is skipped if the behavior b is None.
        """

        b = self.behavior(y)

        if b is None:
            return

        if self.m.get(b) is None:
            self.m[b] = y
        else:
            x = self.m[b]
            if self.is_better(y, x):
                self.m[b] = y

        if self.is_better(self.y, self.best_individual):
            self.best_individual = self.y

    def pipeline(self, problem: Problem):
        self.y = self.variation(self.mutation, self.crossover)
        self.evaluate(problem)
        return self.best_individual

    def variation(self, mutation: Callable, crossover: Callable):
        if random.random() <= self.cx_rate:
            x1 = self.selection()
            x2 = self.selection()
            y = crossover(x1, x2)
        else:
            x = self.selection()
            y = self.clone(x)
            mutation(self.genome(y))
        return y
