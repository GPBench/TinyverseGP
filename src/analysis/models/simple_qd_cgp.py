import copy
import random
from dataclasses import dataclass
from typing import override
from src.analysis.models.simple_cgp import SimpleCGP, SimpleCGPHyperparameters
from src.analysis.models.simple_qd import SimpleQD
from src.gp.tiny_cgp import CGPIndividual, CGPConfig
from src.gp.tinyverse import GPIndividual


@dataclass
class QdCGPHyperparameters(SimpleCGPHyperparameters):
    cx_rate: float

class SimpleQdCGP(SimpleQD, SimpleCGP):
    num_nodes: int

    def __init__(self, functions_: list, terminals_: list, config_: CGPConfig,
                 hyperparameters_: QdCGPHyperparameters):
        SimpleQD.__init__(self, functions_, terminals_, config_, hyperparameters_)
        SimpleCGP.__init__(self, functions_, terminals_, config_, hyperparameters_)

        self.num_nodes = self.hyperparameters.num_function_nodes + self.config.num_inputs
        self.population = None

    def genome(self, x: GPIndividual):
        return x.genome

    def behavior(self, y: GPIndividual):
        return y.genome[-1]

    def clone(self, x: GPIndividual) -> GPIndividual:
        return CGPIndividual(genome_=copy.copy(x.genome))

    @override
    def init(self):
        self.y = self.init_individual()

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
        """
        Evaluates a CGP individual and updates the map.
        Also checks the number of active nodes so that it is not greater than D.
        Returns negative infinity if the genome contains more than D active nodes.
        """

        self.y.fitness = self.penalize(self.evaluate_individual(self.y.genome, problem), self.y.genome)

        if self.best_individual is None:
            self.best_individual = self.y
        self.update(self.y)
        return self.best_individual
