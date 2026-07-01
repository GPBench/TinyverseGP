import operator
from abc import ABC, abstractmethod
from src.gp.tinyverse import GPIndividual

class SimpleQD(ABC):
    m: dict

    def __init__(self):
        self.m = {}

    @abstractmethod
    def update(self, y: GPIndividual):
        pass

    def is_better(self, y, xs: list):
        if self.config.minimizing_fitness:
            comp = operator.gt
        else:
            comp = operator.lt
        for x in xs:
            if comp(y.fitness, x.fitness):
                return False
        return True
