from abc import ABC, abstractmethod
from src.gp.problem import Problem
from src.gp.tinyverse import GPModel, GPHyperparameters
from src.gp.problem import Problem
from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario
import copy
from src.benchmark.symbolic_regression.srbench import SRBench
import numpy as np

class HPOInterface(ABC):
    """
    Abstract class that is used to implement methods for running HPO with GP models
    that are provided within TinverseGP.
    """

    @abstractmethod
    def optimise(self, gpmodel_: GPModel, *params):
        pass


class SMACInterface(HPOInterface):
    def optimise(self, gpmodel_: GPModel, problem : Problem, n_trials_=10) -> GPHyperparameters:
        """
        Runs HPO with SMAC (https://github.com/automl/SMAC3)
        Args:
            gpmodel_ (GPModel): the GP model to be optimised
            n_trials_: Number of trials in the SMAC optimisation environment
            train_X: Training features
            train_y: Training labels
        Returns:
            GPHyperparameters: the hyperparameter configuration to be used
        """
        def train(config: Configuration, seed: int = 0) -> float:
            gpmodel = copy.deepcopy(gpmodel_)
            # Apply hyperparameters
            for c in config.keys():
                setattr(gpmodel.hyperparameters, c, config[c])
            # Use train_X and train_y in the model's training process
            gpmodel.evolve(problem)
            return gpmodel.best_individual.fitness

        # Obtain the hyperparameter (HP) space from the GP model
        paramspace = gpmodel_.hyperparameters.space
        # Use the HP space to init the configuration space (CS)
        configspace = ConfigurationSpace(paramspace)
        # Scenario object specifying the optimization environment
        scenario = Scenario(configspace, deterministic=True, n_trials=n_trials_)
        # Use SMAC to find the best configuration/hyperparameters
        smac = HyperparameterOptimizationFacade(scenario, train)
        incumbent = smac.optimize()

        # Map incumbent to hyperparameters object
        inc_hp = copy.deepcopy(gpmodel_.hyperparameters)
        for c in incumbent.keys():
            setattr(inc_hp, c, incumbent[c])

class SMAC4SRBenchInterface:
    def __init__(self, srbench: SRBench):
        """
        Initialize the SMAC4SRBenchInterface with an SRBench instance.
        Args:
            srbench: An instance of SRBench to be optimized.
        """
        self.srbench = srbench
        self.model_type = srbench.representation  # 'TGP' or 'CGP'

    def optimise(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 10,
        seed: int = 42,
        sc_name: str = "default"
    ) -> GPHyperparameters:
        """
        Optimize the hyperparameters of the SRBench model using SMAC.
        Args:
            X: Training features.
            y: Training labels.
            n_trials: Number of SMAC trials.
            seed: Random seed for reproducibility.
        Returns:
            Dictionary of optimized hyperparameters.
        """

        def train(config: Configuration, seed: int = 0) -> float:
            # Create a copy of the SRBench instance
            srbench = copy.deepcopy(self.srbench)
            # Apply the SMAC configuration to the hyperparameters
            for param, value in config.items():
                setattr(srbench.hyperparameters, param, value)
            # Fit the model
            srbench.fit(X, y)
            # Return the fitness (e.g., negative MSE for minimization)
            return -srbench.score(X, y)  # SMAC minimizes, so return negative score

        # Get the hyperparameter space from the SRBench's hyperparameters
        paramspace = self.srbench.hyperparameters.space
        # Initialize the configuration space
        configspace = ConfigurationSpace(paramspace)
        # Define the SMAC scenario
        output_dir = f"experiments_scripts/smac3-output_{sc_name}_{seed}"
        scenario = Scenario(
            configspace,
            deterministic=True,
            n_trials=n_trials,
            seed=seed,
            output_directory=output_dir
        )
        # Run SMAC optimization
        smac = HyperparameterOptimizationFacade(scenario, train)
        incumbent = smac.optimize()
        inc_hp = copy.deepcopy(self.srbench.hyperparameters)
        for c in incumbent.keys():
            setattr(inc_hp, c, incumbent[c])
        return inc_hp
    
class Hpo:
    """
    Class that provides methods to run HPO for GP models
    that are provided within TinverseGP.
    """

    def optimise_smac(gpmodel_: GPModel, n_trials_=10) -> GPHyperparameters:
        """
        Runs HPO with SMAC.

        Args:
            gpmodel_ (GPModel): the model to be optimised

        Returns:
            GPHyperparameters: the hyperparameter configuration to be used
        """

        def train(config: Configuration, seed: int = 0) -> float:
            gpmodel = copy.deepcopy(gpmodel_)
            for c in config.keys():
                setattr(gpmodel.hyperparameters, c, config[c])
            gpmodel.evolve()
            return gpmodel.best_individual.fitness

        # Obtain the hyperparameter (HP) space from the GP model
        space = gpmodel_.hyperparameters.space

        # Use the HP space to init the configuration space (CS)
        cs = ConfigurationSpace(space)

        # Scenario object specifying the optimization environment
        scenario = Scenario(cs, deterministic=True, n_trials=n_trials_)

        # Use SMAC to find the best configuration/hyperparameters
        smac = HyperparameterOptimizationFacade(scenario, train)
        incumbent = smac.optimize()
        incHP = copy.deepcopy(gpmodel_.hyperparameters)
        for c in incumbent.keys():
            setattr(incHP, c, incumbent[c])
        return incHP
