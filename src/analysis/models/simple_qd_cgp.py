import copy
import random
from dataclasses import dataclass
from typing import override
from src.analysis.models.simple_cgp import SimpleCGP, SimpleCGPConfig
from src.analysis.models.simple_qd import SimpleQD
from src.gp.problem import Problem
from src.gp.tiny_cgp import CGPHyperparameters, CGPIndividual, TinyCGP

@dataclass(kw_only=True)
class QdCGPConfig(SimpleCGPConfig):
    crossover: bool = True

class SimpleQdCGP(SimpleCGP, SimpleQD):
    config: QdCGPConfig
    num_nodes: int

    def __init__(self, functions_: list, terminals_: list, config_: QdCGPConfig,
                 hyperparameters_: CGPHyperparameters):
        SimpleCGP.__init__(self, functions_, terminals_, config_, hyperparameters_)
        SimpleQD.__init__(self)

        self.config = config_
        self.num_nodes = self.hyperparameters.num_function_nodes + self.config.num_inputs

    @override
    def selection(self) -> CGPIndividual:
        if len(self.m) == 0:
            x = random.choice(self.population)
            self.update(x)
        else:
            x = random.choice(list(self.m.values()))
        return x

    def crossover(self, x: list[int]):
        y = copy.copy(x)
        if y[-1] < self.num_nodes - 1:
            out = y[-1] + 1
            pos = self.node_position(out)
        else:
            return CGPIndividual(genome_=y)

        y[pos + 1] = y[-1]
        for i in range(1,self.config.max_arity):
            idx = pos + i + 1
            y[idx] = self.new_value(idx,y[idx])
        y[-1] = out
        return CGPIndividual(genome_=y)

    def update(self, y: CGPIndividual):
        out = y.genome[-1]
        if self.m.get(out) is None:
            self.m[out] = y
        else:
            x = self.m[out]
            if self.is_better(y, [x]):
                self.m[out] = y

    @override
    def pipeline(self, problem: Problem):
        x = self.selection()

        if self.config.crossover:
            y = self.crossover(x.genome)
        else:
            y = CGPIndividual(genome_=copy.copy(x.genome))
        self.mutation(y.genome)
        y.fitness = self.evaluate_individual(y.genome, problem)
        self.update(y)

        return y if self.is_better(y,[x]) else random.choice([x])
