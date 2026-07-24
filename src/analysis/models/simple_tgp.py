"""
Implementation of simple tree-based GP as it has been used for runtime analysis of various
problems.

SimpleTGP uses a (1+1) search strategy and a composite HVL prime mutation operator
consisting of three tree operations: insert, delete and substitute.

A description of SimpleTGP can be found in the work of Neumann et al.
 - https://link.springer.com/chapter/10.1007/978-1-4614-1770-5_7
"""

import copy
import math
import random
from enum import Enum

import numpy as np
from dataclasses import dataclass
from typing import override
from src.gp.tiny_tgp import TGPIndividual, Node, TinyTGP, TGPConfig
from src.gp.tinyverse import Var, Const, Hyperparameters, GPHyperparameters


class HVLPrime:
    """
    This implementation of the HVL prime follows the formal description provided
    in the works of Koetzing et al:
        https://doi.org/10.1145/2330163.2330348
        https://doi.org/10.1016/j.tcs.2013.06.014
    """

    def __init__(self, functions_: list, terminals_: list):
        self.functions = functions_
        self.terminals = terminals_

    def is_leaf(self, n: Node) -> bool:
        """
        Checks whether a given node is a leaf or not.
        """
        return isinstance(n.function, Var) or isinstance(n.function, Const)

    def get_leafs(self, n: Node, p=None, leafs: list = None) -> list[tuple[Node, Node]]:
        """
        Recursively obtain all leafs from the tree and store them in a list.
        """
        if leafs is None:
            leafs = []

        if self.is_leaf(n):
            leafs.append((n, p))
        else:
            for c in n.children:
                self.get_leafs(c, n, leafs)
        return leafs

    def get_inner_nodes(self, n: Node, p=None, inner_nodes: list = None) -> list[tuple[Node, Node]]:
        """
        Recursively obtain all inner nodes from the tree and store them in a list.
        """
        if inner_nodes is None:
            inner_nodes = []

        if not self.is_leaf(n):
            inner_nodes.append((n, p))
            for c in n.children:
                self.get_inner_nodes(c, n, inner_nodes)
        return inner_nodes

    def rnd_inner_node(self, n: Node):
        """
        Select and return a node from the set of inner nodes uniformly at random.
        """
        return random.choice(self.get_inner_nodes(n))

    def rnd_leaf(self, n: Node):
        """
        Select and return a node from the set of leafs uniformly at random.
        """
        return random.choice(self.get_leafs(n))

    def rnd_leaf_dirty(self, n: Node, p=None) -> tuple[Node, Node]:
        """
        by a random depth-first search, no guarantee that the distribution
        is uniform.
        """
        if self.is_leaf(n):
            return n, p
        return self.rnd_leaf_dirty(random.choice(n.children), n)

    def rnd_inner_node_dirty(self, n: Node, p: float) -> Node:
        """
        Return a random inner node by a random depth-first search, no guarantee that the distribution
        is uniform.
        """
        if random.random() <= p:
            return n

        c = [child for child in n.children if self.is_leaf(child) == False]

        if len(c) > 0:
            n = self.rnd_inner_node_dirty(random.choice(c), p)
        return n

    def count_inner_nodes(self, node: Node) -> int:
        """
        Return the number of inner nodes in a tree.
        """
        if len(node.children) == 0:
            return 0
        s = 0
        for child in node.children:
            if child.function is not Var or child.function is not Const:
                s += self.count_inner_nodes(child)
        return 1 + s

    def replace_node(self, u: Node, p: Node):
        """
        Selects a sibling of a given node u by chance and replaces
        u’s parent with the subtree rooted at u’s sibling (hence deleting both u’s parent and the subtree rooted at u from the current tree.

        :param u: selected node
        :param p: parent of u
        """
        cs = [c for c in p.children if c is not u]
        v = random.choice(cs)
        p.function = v.function
        p.children = v.children

    def substitute(self, n: Node):
        """
        Substitute replaces the symbol (function) of a randomly selected inner node of the tree
        with a new function that is selected uniformly at random from the function set.
        """
        n_inner = self.count_inner_nodes(n)

        if n_inner == 0:
            return

        n_rnd = self.rnd_inner_node(n)[0]
        n_rnd.function = random.choice(self.functions)

    def insert(self, n: Node):
        """
        Insert appends an inner node that is uniformly selected at random at the position of a randomly selected leaf
        . The selected leaf as well an additional randomly selected leaf node are then appended as children.
        """
        v, p = self.rnd_leaf_dirty(n)
        u = random.choice(self.terminals)
        w = random.choice(self.functions)

        tmp = v.function
        v.function = w
        v.children.append(Node(function=u, children=[]))
        v.children.append(Node(function=tmp, children=[]))

    def delete(self, n: Node):
        """
        Delete randomly selects a leaf node and it then replaces the parent node with the other child node.
        In this way, the prior selected leaf node as well as the parent node are deleted from
        the tree. Please note that the replacement can be both leaf as well as inner node.
        """
        v, p = self.rnd_leaf(n)

        if p is None:
            return

        self.replace_node(v, p)

    def as_list(self):
        return [self.substitute, self.insert, self.delete]


class NodeUnbiasedHVL(HVLPrime):

    def get_nodes(self, n: Node, p=None, nodes: list = None) -> list[tuple[Node, Node]]:
        """
        Recursively obtain all nodes (inner nodes and leafs) from the tree and store them in a list.
        """
        if nodes is None:
            nodes = []

        nodes.append((n, p))

        if not self.is_leaf(n):
            for c in n.children:
                self.get_nodes(c, n, nodes)

        return nodes

    def rnd_node(self, n: Node):
        """
        Select and return a node from the set of inner nodes uniformly at random.
        """
        return random.choice(self.get_nodes(n))

    def size(self, n: Node) -> int:
        """
        Recursively calculate the number of nodes in the tree.
        """
        if n is None:
            return 0
        if self.is_leaf(n):
            return 1
        return 1 + sum(self.size(c) for c in n.children)

    @override
    def delete(self, n: Node):

        if n is None:
            return

        u, p = self.rnd_node(n)

        if p is None:
            return

        self.replace_node(u, p)


class DepthUnbiasedHVL(HVLPrime):

    def height(self, root: Node, d: int = 0):
        """
        Recursively calculates the height or maximum depth of a tree.
        """
        if root is None:
            return -1
        if self.is_leaf(root): 
            return d
        return max([self.height(c) + 1 for c in root.children])

    def get_nodes_at_depth(self, root: Node, depth: int, nodes: list = None, d: int = 0) -> list[tuple[Node, Node]]:
        """
        Recursively selects and returns all nodes at a specified depth.
        """
        if nodes is None:
            nodes = []

        if depth == 0:
            return [(root, None)]

        if d == depth - 1:
            for c in root.children:
                nodes.append((c, root))

        for c in root.children:
            self.get_nodes_at_depth(c, depth, nodes, d + 1)

        return nodes

    def select_node_at_depth(self, n: Node, d: int) -> tuple[Node, Node]:
        """
        Selects and returns a node at a specified depth uniformly at random.
        """
        nodes = self.get_nodes_at_depth(n, d)
        return random.choice(nodes) if len(nodes) > 0 else (None, None)

    @override
    def delete(self, n: Node):

        h = self.height(n)

        if h == 0:
            return
        else:
            r = random.randint(0, h)

        u, p = self.select_node_at_depth(n, r)

        if p is None:
            return
        else:
            self.replace_node(u, p)

class MutationType(Enum):
    """
    Used for the selection of the mutation method.
    """
    HVL_STD = 0
    HVL_NODE_UNBIASED = 1
    HVL_DEPTH_UNBIASED = 2


@dataclass(kw_only=True)
class SimpleTGPHyperparameters(Hyperparameters):
    """
    Set of hyperparameters used to configure simple TGP.
    """
    lmbda: int = 1
    k: int = 1
    max_depth: int
    check_size: bool = True
    strict_selection: bool = False
    multi: bool = False
    mutation_type: MutationType = MutationType.HVL_STD
    erc = False

    def max_size(self):
        return math.pow(2, self.max_depth + 1) - 1


class SimpleTGP(TinyTGP):
    """
    Simple tree based GP model that is commonly used for runtime analysis.

    Uses (1+1) search strategy and the HVL prime mutation.

    Derives from TinyTGP within TinyverseGP which represent the conventional "vanilla"
    TGP model. Key methods such as initialisation, mutation, breeding and the pipeline
    from TinyTGP are overwritten  to simplify aspects of the standard TGP model.
    """
    hyperparameters: SimpleTGPHyperparameters

    def __init__(self, functions_: list, terminals_: list, config_: TGPConfig,
                 hyperparameters_: SimpleTGPHyperparameters):
        super().__init__(functions_, terminals_, config_, hyperparameters_)

        match self.hyperparameters.mutation_type:
            case MutationType.HVL_STD:
                self.hvl_prime = HVLPrime(functions_, terminals_).as_list()
            case MutationType.HVL_NODE_UNBIASED:
                self.hvl_prime = NodeUnbiasedHVL(functions_, terminals_).as_list()
            case MutationType.HVL_DEPTH_UNBIASED:
                self.hvl_prime = DepthUnbiasedHVL(functions_, terminals_).as_list()

    @override
    def init(self):
        """
        Initialises the population.
        """
        self.population = [self.init_individual() for _ in range(self.hyperparameters.lmbda + 1)]

    def init_individual(self) -> TGPIndividual:
        """
        Initialises an individual with a genome
        """
        return TGPIndividual(genome_=[self.init_tree_simple()])

    def init_tree_simple(self):
        """
        Simplified version of the tree init method that only creates tree
        with one leaf uniformly selected at random.
        """
        return Node(function=random.choice(self.terminals), children=[])

    @override
    def perturb(self, parent1: Node, parent2: Node = None) -> Node:
        return self.mutation(parent1)

    @override
    def mutation(self, parent: Node) -> Node:
        """
        Overrides the mutation method of TinyTGP which is by default the subtree
        mutation.

        The mutation method of SimpleTGP either performs HVL-single or the multi-strategy.

        HVL-single only perform one operation in the framework of the mutation procedure
        while HVL-multi perform k steps that are drawn from poisson distribution.

        """
        if self.hyperparameters.multi:
            k = 1 + np.random.poisson(1)
        else:
            k = self.hyperparameters.k

        for _ in range(k):
            random.choice(self.hvl_prime)(parent)
        return parent

    @override
    def breed(self):
        """
        Breeding procedure that first selects the parent according to the chosen selection
        strategy. Then the parent individual is cloned and mutated. Depending on the choice of
        the size check option, the offspring is only replaces the parent when its size is less or
        equal the maximum tree size.
        """
        parent = self.selection()
        self.population = [parent]
        for _ in range(self.hyperparameters.lmbda):
            genome = copy.deepcopy(parent.genome[0])
            genome = [self.perturb(genome)]

            if self.hyperparameters.check_size:
                if self.eval_complexity(genome) > self.hyperparameters.max_size():
                    genome = [copy.deepcopy(parent.genome[0])]

            offspring = TGPIndividual(
                genome_=genome
            )
            self.population.append(offspring)

    @override
    def selection(self) -> TGPIndividual:
        """
        Selects the parent from the population according to the chosen selection strategy
        - non-strict (also known as random local search RLS) or strict selection.
        """
        sorted_pop = sorted(
            self.population,
            key=lambda ind: ind.fitness,
            reverse=not self.config.minimizing_fitness,
        )
        count = 0
        if not self.hyperparameters.strict_selection:
            best_fitness = sorted_pop[0].fitness
            for individual in sorted_pop:
                if individual.fitness != best_fitness:
                    break
                else:
                    count += 1
            parent = random.randint(0, count - 1)
        else:
            parent = 0
        return sorted_pop[parent]

    @override
    def pipeline(self, problem):
        """
        Pipeline of simple TGP:
         -> Selection -> Mutation -> Evaluation
        """
        self.breed()
        return self.evaluate(problem)
