from abc import ABC, abstractmethod
from src.gp.tinyverse import GPIndividual

class SimpleQD(ABC):
    m: dict

    def __init__(self):
        self.m = {}

    @abstractmethod
    def update(self, y: GPIndividual):
        pass
