import copy
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import override
from src.analysis.models.simple_qd import SimpleQD
from src.analysis.models.simple_tgp import SimpleTGP, SimpleTGPHyperparameters, InitMethod, SimpleTGPConfig
from src.gp.tiny_tgp import TGPIndividual, Node
from src.gp.tinyverse import GPConfig, GPIndividual



@dataclass
class QdTGPHyperparameters(SimpleTGPHyperparameters):
    cx_rate: float


class SimpleQdTGP(SimpleQD, SimpleTGP):
    config: SimpleTGPConfig
    xs: list[GPIndividual]

    def __init__(self, functions_: list, terminals_: list, config_: SimpleTGPConfig,
                 hyperparameters_: QdTGPHyperparameters):
        SimpleQD.__init__(self, functions_, terminals_, config_, hyperparameters_)
        SimpleTGP.__init__(self, functions_, terminals_, config_, hyperparameters_)
        self.xs = []
        self.population = None

    def genome(self, x: GPIndividual):
        return x.genome[0]

    def behavior(self, y: GPIndividual):
        """
        Determines the behavior that is defined as the height of the tree:
            - Returns None if max depth is already reached
            - The height is negative (-1) for empty programs (tree's)
        """
        return self.height(y.genome[0]) if self.is_valid(y.genome) else None

    def clone(self, x: GPIndividual) -> GPIndividual:
        return TGPIndividual(genome_=copy.deepcopy(x.genome))

    @override
    def init(self):
        self.y = self.init_individual()

    @override
    def crossover(self, x1: TGPIndividual, x2: TGPIndividual) -> TGPIndividual:
        self.xs.append(x1)
        self.xs.append(x2)
        g1, g2 = x1.genome[0], x2.genome[0]
        n = Node(function=random.choice(self.functions), children=[])
        n.children.append(copy.deepcopy(g1))
        n.children.append(copy.deepcopy(g2))

        if self.hyperparameters.check_complexity:
            if self.eval_complexity([n]) > self.hyperparameters.max_size():
                return random.choice([x1, x2])

        return TGPIndividual(genome_=[n], fitness_=None)

    @override
    def evaluate(self, problem) -> GPIndividual:
        """
        Evaluates the fitness of TGP individual and updates the map.

        Validity checks are performed with the penalize functions from the tinyverse
        module:
            - If height is greater the maximum height the penalize function
                returns negative infinity.
        """

        if self.y.fitness is not None:
            return self.y

        if len(self.xs) == 0:
            self.y.fitness = self.penalize(self.evaluate_individual(self.y.genome, problem), self.y.genome)
        else:
            self.y.fitness = self.penalize(self.y.genome[0].function(self.xs[0].fitness,
                                                       self.xs[1].fitness), self.y.genome)
            self.xs.clear()

        self.update(self.y)

        return self.best_individual
