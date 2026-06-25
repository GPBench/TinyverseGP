import copy
import random
from typing import override
from src.analysis.models.simple_cgp import SimpleCGP, SimpleCGPConfig
from src.analysis.models.simple_qd import SimpleQD
from src.gp.problem import Problem
from src.gp.tiny_cgp import CGPHyperparameters, CGPIndividual


class SimpleQdCGP(SimpleCGP, SimpleQD):

    def __init__(self, functions_: list, terminals_: list, config_: SimpleCGPConfig,
                 hyperparameters_: CGPHyperparameters):
        SimpleCGP.__init__(self,functions_, terminals_, config_, hyperparameters_)
        SimpleQD.__init__(self)

    def is_better(self, ind1, ind2):
        return ind1.fitness <= ind2.fitness if self.config.minimizing_fitness \
            else ind1.fitness >= ind2.fitness

    def update(self, y: CGPIndividual):
        out = y.genome[-1]
        if self.m.get(out) is None:
            self.m[out] = y
        else:
            x = self.m[out]
            if self.is_better(y, x):
                self.m[out] = y

    @override
    def pipeline(self, problem: Problem):
        if len(self.m) == 0:
            x = random.choice(self.population)
            self.update(x)
        else:
            x = random.choice(list(self.m.values()))

        y = CGPIndividual(genome_=copy.copy(x.genome))
        self.mutation(y.genome) # Perform SAM
        y.fitness = self.evaluate_individual(y.genome, problem)
        self.update(y)

        return y if problem.is_better(y.fitness, x.fitness) else x
