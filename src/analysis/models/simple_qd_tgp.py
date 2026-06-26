import copy
import random
from typing import override

from src.analysis.models.simple_qd import SimpleQD
from src.analysis.models.simple_tgp import SimpleTGP, SimpleTGPHyperparameters
from src.gp.problem import Problem
from src.gp.tiny_tgp import TGPIndividual, TGPConfig, Node


class SimpleQdTGP(SimpleTGP, SimpleQD):

    def __init__(self, functions_: list, terminals_: list, config_: TGPConfig,
                 hyperparameters_: SimpleTGPHyperparameters):
        SimpleTGP.__init__(self,functions_, terminals_, config_, hyperparameters_)
        SimpleQD.__init__(self)

    def height(self, root: Node, d:int=0):
        if root.function in self.terminals:
            return d
        else:
            depths = []
            for child in root.children:
                depths.append(self.height(child, d + 1))
            return max(depths)

    def is_better(self, ind1, ind2):
        return ind1.fitness <= ind2.fitness if self.config.minimizing_fitness \
            else ind1.fitness >= ind2.fitness

    def update(self, y: TGPIndividual):
        max_depth = self.height(y.genome[0])
        if self.m.get(max_depth) is None:
            self.m[max_depth] = y
        else:
            x = self.m[max_depth]
            if self.is_better(y, x):
                self.m[max_depth] = y

    @override
    def pipeline(self, problem: Problem):
        if len(self.m) == 0:
            x = random.choice(self.population)
            self.update(x)
        else:
            x = random.choice(list(self.m.values()))

        y = TGPIndividual(genome_=copy.deepcopy(x.genome))
        self.mutation(y.genome[0])
        y.fitness = self.evaluate_individual(y.genome, problem)
        self.update(y)

        return y if problem.is_better(y.fitness, x.fitness) else x
