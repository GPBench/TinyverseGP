"""
Provides an implementation of the problems AND_n and XOR_n, two benchmarks used for runtime analysis
of GP algorithms in evolving conjunctions and disjunctions.

The dataset used for learning the respective Boolean function is based on its truth table representation.

The base problem class provides three different configurations for defining training and testing set:
  - complete: The full dataset (truth table) is used and therefore the exact solution
                should be synthesised.
  - split: Standard train-test-split strategy of the dataset based on predefined split proportions
            for train and test data. Used for testing the generalisation abilities of a GP method.
  - sample: Incomplete training and testing sets are sampled by generating observations uniformly at randon.
            Used for testing the generalisation abilities of a GP method for large dimensions since
            the full (exponentially growing) dataset does not need to be generated.

The generalisation error can be calculated in two different ways:
    - exact: testing the program on the complete dataset
    - approximate: testing the program only on a sample of the dataset
"""

import random
from enum import Enum
from math import pow, ceil
from dataclasses import dataclass
from typing_extensions import override, Any
from sklearn.model_selection import train_test_split
from src.gp.loss import hamming_distance_bitwise
from src.gp.problem import BlackBox
from functools import reduce
import numpy as np

from src.gp.tinyverse import GPModel


class ErrorType(Enum):
    EXACT = 0
    APPROXIMATE = 1


class DatasetType(Enum):
    COMPLETE = 0
    SPLIT = 1
    SAMPLE = 2


@dataclass
class Dataset:
    """
    Representation of the dataset based which stores the input-output
    matching of the Boolean function.

    The variables and outputs of the truth table are represented with
    pseudo-Boolean values.

    Attributes:
        data : aray containing the
        n_rows : number of rows of the dataset
        n_cols : number of cols of the dataset
        X : set of the observations (variables) of the truth table
        y : set of outputs of the Boolean function
    """
    data: np.array
    n_rows: int = None
    n_cols: int = None
    X: np.array = None
    y: np.array = None

    def __len__(self):
        return self.n_rows

    def __post_init__(self):
        """
        Post-initialisation procedures determines the dimensions of the
        data and splits the dataset into observations and actual values.
        """
        if self.data is None:
            return

        if len(self.data) == 0:
            return

        if self.n_rows is None:
            self.n_rows = len(self.data)

        if self.n_cols is None:
            self.n_cols = len(self.data[0])

        if self.X is None:
            self.X = self.data[:, : self.n_cols - 1]

        if self.y is None:
            self.y = self.data[:, self.n_cols - 1]

    def get_observations(self):
        return self.X

    def get_actual(self):
        return self.y


class BooleanFunction(BlackBox):
    """

    Implements a problem scenario for Boolean function learning based on TinverseGP's
    black box scenario.

    Provides methods for creating the datasets in three different ways (see description
    above).

    Attributes:
        n_in: number of inputs
        n_out: number of outputs
        dataset: attribute for storing the unsplit dataset
        train: training set object
        test: testing set object
        train_size: dimension of the training set
        testing_size: dimension of the testing set
        operator: Boolean operator that is used to calculate the input-output matchings
    """

    n_in: int
    n_out: int
    dataset: Dataset
    train: Dataset
    test: Dataset
    train_size: int
    test_size: int
    operator: callable
    k: float

    def __init__(self, n_in: int, n_out: int, operator: callable,
                 negated_vars=False, k: float = 1.3, dataset_type: DatasetType = DatasetType.COMPLETE,
                 error_type: ErrorType = ErrorType.APPROXIMATE):

        super().__init__(actual_=[0],
                         observations_=[0],
                         loss_=hamming_distance_bitwise,
                         ideal_=0,
                         minimizing_=True)

        # Init the complete dataset only if necessary, either when a complete training set should be
        # used or it should be split into train/test proportions
        if dataset_type == DatasetType.COMPLETE or dataset_type == DatasetType.SPLIT:
            self.dataset = self.init_dataset(n_in, n_out, operator, negated_vars)

        self.n_in = n_in
        self.n_out = n_out
        self.operator = operator
        self.negated_vars = negated_vars
        self.k = k

        # Define the size of the training set based on the selected scenario
        # i.e. using complete or incomplete training set
        if dataset_type == DatasetType.SPLIT or dataset_type == DatasetType.SAMPLE:
            assert k is not None
            self.train_size = ceil(pow(self.n_in, self.k))
        else:
            self.train_size = self.dataset.n_rows

        # Setup the training and testing set based on the chosen dataset scenario
        match dataset_type:
            case DatasetType.COMPLETE:
                self.train = self.test = self.dataset
            case DatasetType.SPLIT:
                self.test_size = self.dataset.n_rows - self.train_size
                self.train, self.test = self.split_dataset()

                # Consider this subcase if the exact error should be determined
                if error_type == ErrorType.EXACT:
                    self.test = self.dataset
            case DatasetType.SAMPLE:
                self.test_size = self.train_size
                self.train = self.sample_dataset(self.train_size, n_in, self.operator)
                self.test = self.sample_dataset(self.test_size, n_in, self.operator)

        # Init the observation and actual value set of the base black-box class
        self.observations, self.actual = self.train.get_observations(), self.train.get_actual()


    def init_variables(self, n_in: int, n_out: int, negated_vars: bool = False) -> tuple:
        """
        Initialises the input variables used to create the complete dataset (truth table).
        If desired the method also add negated values of the input variables to increase problem hardness.

        :param n_in: number of inputs
        :param n_out: number of outputs
        :param negated_vars: status if negated inputs values should be added
        :return n_rows, n_cols, dataset: dimensions of the dataset and the dataset array
                                            with input values
        """

        # Calculate the dimensions of the dataset by considering the setting whether
        # negated variables should be used
        n_rows = int(pow(2, n_in))
        if not negated_vars:
            n_cols = n_in + n_out
        else:
            n_cols = (2 * n_in) + n_out

        # Define number of input variables and instantiate the dataset
        n_vars = n_in
        dataset = np.zeros(shape=(n_rows, n_cols), dtype=np.int8)

        # Create the input table for each variable and row
        for c in range(n_vars):
            d = pow(2, n_vars - c)
            for r in range(n_rows):
                if r % d >= d / 2:
                    dataset[r][c] = 1

        # Add negated vars for each row
        if negated_vars:
            for r in range(n_rows):
                for c in range(n_vars):
                    dataset[r][n_vars + c] = 1 if dataset[r][c] == 0 else 0

        return n_rows, n_cols, dataset

    def init_dataset(self, n_in: int, n_out: int, operator: callable, negated_vars: bool = False) -> Dataset:
        """
        Creates the dataset object including the data array that contains the truth table of the function

        :param n_in: number of inputs
        :param n_out: number of outputs
        :param negated_vars: status if negated inputs values should be added
        :param operator: Boolean operator that is used to calculate the input-output matchings
        :return dataset: Dataset object
        """

        # Generate the table for the input variables first
        n_rows, n_cols, dataset = self.init_variables(n_in, n_out, negated_vars)

        # Calculate the actual output value row-wise
        for row in dataset:
            # Get the subset of variables
            args = row[0:n_in]
            # Perform the calculation of the output value
            res = reduce(operator, args)
            row[n_cols - n_out] = res

        return Dataset(data=dataset, n_cols=n_cols, n_rows=n_rows)

    def sample_observation(self, n_in: int, negated_vars:bool=False) -> np.array:
        """
        Samples one observation uniformly at random.

        :param n_in: number of inputs
        :param negated_vars: status if negated inputs values should be added
        :return obs: sampled observation
        """
        if not negated_vars:
            n_vars = n_in
        else:
            n_vars = 2 * n_in

        obs = []
        for _ in range(n_vars):
            val = 0 if random.random() < 0.5 else 1
            obs.append(val)
        return obs

    def sample_dataset(self, size: int, n_in: int, operator: callable, negated_vars:bool=False) -> np.array:
        """
        Samples a dataset of given size

        @parm size: size of the sampled set
        :param n_in: number of inputs
        :param operator: Boolean operator that is used to calculate the input-output matchings
        :param negated_vars: status if negated inputs values should be added
        """
        data = []
        for _ in range(size):
            obs = self.sample_observation(n_in, negated_vars=False)
            if negated_vars:
                args = obs[0:n_in]
            else:
                args = obs
            act = reduce(operator, args)
            row = obs + [act]
            data.append(row)
        return Dataset(data=np.array(data))

    def split_dataset(self) -> tuple:
        """
        Performs a common train-test split by using the train_test_split method from sklearn.
        Stacks the training and testing data then accordingly to form subtable that used for
        evaluation.

        :return: Dataset objects for training and testing set
        """
        assert self.dataset is not None
        test_frac = self.test_size / self.dataset.n_rows
        X_train, X_test, y_train, y_test = train_test_split(
            self.dataset.X, self.dataset.y, test_size=test_frac)
        train_data = np.column_stack((X_train, y_train))
        test_data = np.column_stack((X_test, y_test))
        return Dataset(data=train_data), Dataset(data=test_data)

    def calc_generalisation_error(self, program: Any, model: GPModel) -> float:
        """
        Calculates the generalisation error by evaluating the best program found on the test set.
        Depending on the dataset scenario the error is either exact or approximated.

        :param program: best program found so far
        :param model: the gp model that is used for evaluation
        """
        self.observations, self.actual = self.test.get_observations(), self.test.get_actual()
        return self.evaluate(program, model)

    @override
    def cost(self, predictions: list) -> float:
        return super().cost(predictions)


class Conjunction(BooleanFunction):
    """
    Derived class to represent logical conjunction (AND) as benchmark problem.
    """

    def __init__(self, n: int, negated_vars: bool = False, k: int = 1.3,
                 dataset_type: DatasetType = DatasetType.COMPLETE, error_type: ErrorType = ErrorType.EXACT):
        super().__init__(n_in=n, n_out=1, operator=lambda x, y: x & y,
                         negated_vars=negated_vars,
                         k=k,
                         dataset_type=dataset_type)


class ExclusiveDisjunction(BooleanFunction):
    """
    Derived class to represent logical exclusive disjunction (XOR) as benchmark problem.
    """

    def __init__(self, n: int, negated_vars: bool = False, k: int = 1.3, dataset_type=DatasetType.COMPLETE,
                 error_type: ErrorType = ErrorType.EXACT):
        super().__init__(n_in=n, n_out=1, operator=lambda x, y: x ^ y,
                         negated_vars=negated_vars,
                         k=k,
                         dataset_type=dataset_type,
                         error_type=error_type)
