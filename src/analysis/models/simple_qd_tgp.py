import copy
import operator
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

    def is_better(self, y, xs: list):
        if self.config.minimizing_fitness:
            comp = operator.gt
        else:
            comp = operator.lt
        for x in xs:
            if comp(y.fitness, x.fitness):
                return False
        return True

    def update(self, y: TGPIndividual):
        max_depth = self.height(y.genome[0])
        if self.m.get(max_depth) is None:
            self.m[max_depth] = y
        else:
            x = self.m[max_depth]
            if self.is_better(y, [x]):
                self.m[max_depth] = y

    @override
    def selection(self) -> TGPIndividual:
        if len(self.m) == 0:
            x = random.choice(self.population)
            self.update(x)
        else:
            x = random.choice(list(self.m.values()))
        return x

    @override
    def crossover(self, x1: Node, x2: Node) -> TGPIndividual:
        n = Node(function=random.choice(self.functions), children=[])
        n.children.append(x1)
        n.children.append(x2)
        return TGPIndividual(genome_=[n])

    @override
    def pipeline(self, problem: Problem):
        x1 = self.selection()
        x2 = self.selection()
        y = self.crossover(x1.genome[0], x2.genome[0])
        y.fitness = y.genome[0].function(x1.fitness,
                                         x2.fitness)

        self.update(y)

        return y if self.is_better(y,[x1,x2]) else random.choice([x1,x2])
