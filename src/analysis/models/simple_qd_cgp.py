import copy
import random
from dataclasses import dataclass
from typing import override
from src.analysis.models.simple_cgp import SimpleCGP
from src.analysis.models.simple_qd import SimpleQD
from src.gp.problem import Problem
from src.gp.tiny_cgp import CGPHyperparameters, CGPIndividual, CGPConfig
from src.gp.tinyverse import GPIndividual


@dataclass
class QdCGPHyperparameters(CGPHyperparameters):
    cx_rate: float

class SimpleQdCGP(SimpleQD, SimpleCGP):
    num_nodes: int
    xs: list

    def __init__(self, functions_: list, terminals_: list, config_: CGPConfig,
                 hyperparameters_: QdCGPHyperparameters):
        SimpleQD.__init__(self)
        SimpleCGP.__init__(self, functions_, terminals_, config_, hyperparameters_)

        self.config = config_
        self.num_nodes = self.hyperparameters.num_function_nodes + self.config.num_inputs
        self.xs = []
        self.y = None

    def genome(self, x: GPIndividual):
        return x.genome

    def behavior(self, y: GPIndividual):
        return y.genome[-1]

    def clone(self, x: GPIndividual) -> GPIndividual:
        return CGPIndividual(genome_=copy.copy(x.genome))

    @override
    def init(self):
        self.y = self.init_individual()

    @override
    def selection(self) -> CGPIndividual:
        if len(self.m) == 0:
            x = random.choice(self.population)
            self.update(x)
        else:
            x = random.choice(list(self.m.values()))
        return x

    def recombine(self, x: list[int], clone=False):
        if clone:
            y = copy.copy(x)
        else:
            y = x

        if y[-1] < self.num_nodes - 1:
            out = y[-1] + 1
            pos = self.node_position(out)
        else:
            return y

        y[pos + 1] = y[-1]
        for i in range(1,self.config.max_arity):
            idx = pos + i + 1
            y[idx] = self.new_value(idx,y[idx])
        y[-1] = out
        return y

    def crossover(self, x1: CGPIndividual, x2: CGPIndividual, recombine=True):
        x1, x2 = x1.genome, x2.genome
        cx_node = random.randint(1, self.num_nodes)
        cx_pos = self.node_position(cx_node)
        y = x1[:cx_pos] + x2[cx_pos:]
        if recombine:
            y = self.recombine(y)
        return CGPIndividual(genome_=y)

    @override
    def evaluate(self, problem) -> GPIndividual:
        self.y.fitness = self.evaluate_individual(self.y.genome, problem)
        self.update(self.y)
        return self.y

    @override
    def pipeline(self, problem: Problem):
        if self.y.fitness is None:
            self.evaluate(problem)
            self.update(self.y)
        self.y = self.variation(self.mutation, self.crossover)
        self.evaluate(problem)
        return self.y if self.is_better(self.y, self.xs) else random.choice(self.xs)
