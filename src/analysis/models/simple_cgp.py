"""
Implementation of simple Cartesian Genetic Programming (CGP) as it has been
used for runtime analysis.

SimpleCGP uses a (1+1) search strategy and either probabilistic or
single-active-gene mutation for genetic variation.
"""

import random
from dataclasses import dataclass
from enum import Enum

from typing_extensions import override
from src.gp.tiny_cgp import CGPHyperparameters, CGPConfig, TinyCGP

@dataclass(kw_only=True)
class SimpleCGPHyperparameters(CGPHyperparameters):
    """
    Set of hyperparameters used to configure simple TGP.
    """
    max_active_nodes: int

class MutationType(Enum):
    """
    Used for the selection of the mutation method.
        - PROB: probabilistic mutation
        - SAM: single-active-gene mutation
    """
    PROB = 0
    SAM = 1

@dataclass(kw_only=True)
class SimpleCGPConfig(CGPConfig):
    """
    Simple CGP config that has been derived from the config used for TinyCGP and
    is extended by providing an option for the mutation type (either SAM or PROB).
    """
    log_scaling: bool = False
    mutation_type: MutationType = MutationType.SAM

class SimpleCGP(TinyCGP):
    """
    Simple CGP model derived from TinyCGP. Uses a (1+1) search strategy either with
    strict or non-strict selection. That search strategy and selection mechanism is
    already integrated in TinyCGP.

    For the implementation key methods such as mutation from TinyCGP are overwritten
    to simplify aspects of the standard model.
    """
    config: SimpleCGPConfig
    hyperparameters: SimpleCGPHyperparameters

    def __init__(self, functions_: list, terminals_: list, config_: SimpleCGPConfig, hyperparameters_: SimpleCGPHyperparameters):
        super().__init__(functions_, terminals_, config_, hyperparameters_)
        self.config = config_
        self.hyperparameters.lmbda = 1

    def new_value(self, position: int, old_val: int=None) -> int:
        """
        Returns a new gene value that is uniformly selected at random and respects
        the type of the gene (function, output or connection gene) as well as the position
        in genotype.

        When an old_value is given, this value is then excluded from the set of possible values
        for the new gene value.
        """
        gene_type = self.phenotype(position)

        # Init the set of excluded values for the sampling of the new value
        if old_val is None:
            exclude = []
        else:
            exclude = [old_val]

        # Calculate the bounds for sampling a new gene value
        # depending on the gene type
        if gene_type == self.GeneType.CONNECTION:
            levels_back = self.hyperparameters.levels_back
            node_num = self.node_number(position)
            if node_num <= levels_back:
                min = 0
            else:
                min = node_num - levels_back
            max = node_num - 1
        elif gene_type == self.GeneType.FUNCTION:
            min = 0
            max = self.config.num_functions - 1
        else: # In this case it is an output gene
            min = 0
            max = self.config.num_inputs + self.hyperparameters.num_function_nodes - 1

        # Sample new potential gene values uniformly at random
        # w.r.t. the bounds and excluded values
        new_vals = [c for c in range(min, max + 1) if c not in exclude]

        # Select a new gene value by chance if it is possible
        # otherwise keep the old value
        return random.choice(new_vals) if len(new_vals) > 0 else old_val

    def mutation_prob(self, genome: list[int]):
        """
        Perform the standard probabilistic mutation sets perform a mutation
        by chance w.r.t to a predefined mutation rate.
        """
        for idx in range(len(genome)):
            if random.random() < self.hyperparameters.mutation_rate:
                gene_val = genome[idx]
                genome[idx] =self.new_value(idx, old_val=gene_val)

    def mutation_sam(self, genome: list[int]):
        """
        Single-active-gene mutation (SAM) enforces the mutation of a single active
        gene in the genotype.
        """
        active_nodes = self.active_nodes(genome)
        active = False
        while not active:
            # Randomly select and mutate one gene per iteration until
            # an active gene was altered
            gene_pos = random.randint(0, self.config.num_genes - 1)
            gene_val = genome[gene_pos]
            genome[gene_pos] = self.new_value(gene_pos, old_val=gene_val)
            node_num = self.node_number(gene_pos)

            # Check whether the selected gene is active (output genes are considered
            if node_num in active_nodes or self.phenotype(gene_pos) == self.GeneType.OUTPUT:
                active = True

    @override
    def is_valid(self, genome: list[int]) -> bool:
        """
        """
        return self.eval_complexity(genome) <= self.hyperparameters.max_active_nodes

    @override
    def mutation(self, genome: list[int]):
        """
        Calls the pre-selected mutation type (either SAM or PROB).
        """
        if self.config.mutation_type == MutationType.SAM:
            self.mutation_sam(genome)
        else:
            self.mutation_prob(genome)


