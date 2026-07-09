import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import override
from src.analysis.models.simple_qd import SimpleQD
from src.analysis.models.simple_tgp import SimpleTGP, SimpleTGPHyperparameters
from src.gp.problem import Problem
from src.gp.tiny_tgp import TGPIndividual, Node
from src.gp.tinyverse import GPConfig, GPIndividual


class InitMethod(Enum):
    MIN = 0
    GROW = 1
    FULL = 2


@dataclass(kw_only=True)
class QdTGPConfig(GPConfig):
    mutation: bool = True
    init_method: InitMethod = InitMethod.MIN


@dataclass(kw_only=True)
class QdTGPHyperparameters(SimpleTGPHyperparameters):
    erc: bool = False


class SimpleQdTGP(SimpleTGP, SimpleQD):
    xs: list
    y: TGPIndividual
    config: QdTGPConfig

    def __init__(self, functions_: list, terminals_: list, config_: QdTGPConfig,
                 hyperparameters_: SimpleTGPHyperparameters):
        SimpleQD.__init__(self)
        SimpleTGP.__init__(self, functions_, terminals_, config_, hyperparameters_)
        self.xs = []
        self.y = None

    @override
    def init(self):
        self.y = self.init_individual()

    @override
    def init_individual(self) -> TGPIndividual:
        """
        Initializes an individual with a genome
        """
        if self.config.init_method == InitMethod.MIN:
            return TGPIndividual(genome_=[self.init_tree_simple()])
        elif self.config.init_method == InitMethod.GROW:
            return TGPIndividual(genome_=[self.tree_random_grow(min_depth=1, max_depth=self.hyperparameters.max_depth,
                                                                size=self.hyperparameters.max_size())])
        else:
            return TGPIndividual(genome_=[self.tree_random_full(max_depth=self.hyperparameters.max_depth,
                                                                size=self.hyperparameters.max_size())])

    def height(self, root: Node, d: int = 0):
        if root.function in self.terminals:
            return d
        else:
            left = self.height(root.children[0], d + 1)
            right = self.height(root.children[1], d + 1)
            return max(left, right)

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
        return random.choice(list(self.m.values()))

    @override
    def crossover(self, x1: Node, x2: Node) -> TGPIndividual:
        n = Node(function=random.choice(self.functions), children=[])
        n.children.append(x1)
        n.children.append(x2)
        return TGPIndividual(genome_=[n], fitness_=None)

    @override
    def evaluate(self, problem) -> GPIndividual:
        if self.y.fitness is not None:
            return self.y

        if self.config.mutation or len(self.xs) == 0:
            self.y.fitness = self.evaluate_individual(self.y.genome, problem)
        else:
            self.y.fitness = self.y.genome[0].function(self.xs[0].fitness,
                                                       self.xs[1].fitness)
        self.update(self.y)

        return self.y

    @override
    def breed(self):
        self.xs.clear()

        x1 = self.selection()
        x2 = self.selection()

        self.xs.append(x1)
        self.xs.append(x2)

        y = self.crossover(x1.genome[0], x2.genome[0])

        if self.config.mutation:
            y.genome[0] = copy.deepcopy(y.genome[0])
            self.mutation(y.genome[0])

        return y

    @override
    def pipeline(self, problem: Problem):
        self.y = self.breed()
        self.evaluate(problem)

        return self.y if self.is_better(self.y, self.xs) else random.choice(self.xs)
