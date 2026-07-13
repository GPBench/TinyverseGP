import operator
import random
from abc import ABC, abstractmethod
from src.gp.tinyverse import GPIndividual, GPConfig, Hyperparameters


class SimpleQD(ABC):
    m: dict
    xs: list[GPIndividual]
    y: GPIndividual
    cx_rate: float

    def __init__(self, cx_rate_: float = 0.5):
        self.m = {}
        self.cx_rate = cx_rate_

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

    def update(self, y: GPIndividual):
        b = self.behavior(y)
        if self.m.get(b) is None:
            self.m[b] = y
        else:
            x = self.m[b]
            if self.is_better(y, [x]):
                self.m[b] = y

    def is_better(self, y, xs: list[GPIndividual]):
        if self.config.minimizing_fitness:
            comp = operator.gt
        else:
            comp = operator.lt
        for x in xs:
            if comp(y.fitness, x.fitness):
                return False
        return True

    def variation(self, mutation: callable, crossover: callable):
        self.xs.clear()
        if random.random() <= self.cx_rate:
            x1 = self.selection()
            self.xs.append(x1)
            x2 = self.selection()
            self.xs.append(x2)
            y = crossover(x1, x2)
        else:
            x = self.selection()
            self.xs.append(x)
            y = self.clone(x)
            mutation(self.genome(y))
        return y






