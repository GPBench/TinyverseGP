import random
from enum import Enum
from math import pow, ceil
from dataclasses import dataclass
from typing_extensions import override
from sklearn.model_selection import train_test_split
from src.gp.loss import hamming_distance_bitwise
from src.gp.problem import BlackBox
from functools import reduce
import numpy as np


class DatasetType(Enum):
    COMPLETE = 0
    SPLIT = 1
    SAMPLE = 2


@dataclass
class Dataset:
    data: np.array
    n_rows: int = None
    n_cols: int = None
    X: np.array = None
    y: np.array = None

    def __len__(self):
        return self.n_rows

    def __post_init__(self):
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
    n_in: int
    n_out: int
    dataset: Dataset
    train: Dataset
    test: Dataset
    operator: callable
    train_size: int
    test_size: int
    k: float

    def __init__(self, n_in: int, n_out: int, operator: callable,
                 negated_vars=False, k: float = None, dataset_type: DatasetType = DatasetType.COMPLETE):

        super().__init__(actual_=[0],
                         observations_=[0],
                         loss_=hamming_distance_bitwise,
                         ideal_=0,
                         minimizing_=True)

        if dataset_type == DatasetType.COMPLETE or dataset_type == DatasetType.SPLIT:
            self.dataset = self.init_dataset(n_in, n_out, operator, negated_vars)

        self.n_in = n_in
        self.n_out = n_out
        self.operator = operator
        self.negated_vars = negated_vars
        self.k = k

        if dataset_type == DatasetType.SPLIT or dataset_type == DatasetType.SAMPLE:
            assert k is not None
            self.train_size = ceil(pow(self.n_in, self.k))
        else:
            self.train_size = self.dataset.n_rows

        match dataset_type:
            case DatasetType.COMPLETE:
                self.train = self.test = self.dataset
            case DatasetType.SPLIT:
                self.test_size = self.dataset.n_rows - self.train_size
                self.train, self.test = self.split_dataset()
            case DatasetType.SAMPLE:
                self.test_size = self.train_size
                self.train = self.sample_dataset(self.train_size, n_in, self.operator)
                self.test = self.sample_dataset(self.test_size, n_in, self.operator)

        self.observations, self.actual = self.train.get_observations(), self.train.get_actual()

    def init_variables(self, n_in, n_out, negated_vars=False) -> tuple:
        n_rows = int(pow(2, n_in))
        if not negated_vars:
            n_cols = n_in + 1
        else:
            n_cols = (2 * n_in) + 1
        n_vars = n_in
        dataset = np.zeros(shape=(n_rows, n_cols), dtype=np.int8)

        for c in range(n_vars):
            d = pow(2, n_vars - c)
            for r in range(n_rows):
                if r % d >= d / 2:
                    dataset[r][c] = 1

        if negated_vars:
            for r in range(n_rows):
                for c in range(n_vars):
                    dataset[r][n_vars + c] = 1 if dataset[r][c] == 0 else 0

        return n_rows, n_cols, dataset

    def sample_observation(self, n_in, negated_vars=False) -> np.array:
        if not negated_vars:
            n_vars = n_in
        else:
            n_vars = 2 * n_in

        obs = []
        for _ in range(n_vars):
            val = 0 if random.random() < 0.5 else 1
            obs.append(val)
        return obs

    def sample_dataset(self, size, n_in, operator, negated_vars=False) -> np.array:
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


    def init_dataset(self, n_in, n_out, operator, negated_vars=False) -> Dataset:
        n_rows, n_cols, training_set = self.init_variables(n_in, n_out, negated_vars)

        for row in training_set:
            args = row[0:n_in]
            res = reduce(operator, args)
            row[n_cols - n_out] = res

        return Dataset(data=training_set, n_cols=n_cols, n_rows=n_rows)

    def split_dataset(self) -> tuple:
        assert self.dataset is not None
        test_frac = self.test_size / self.dataset.n_rows
        X_train, X_test, y_train, y_test = train_test_split(
            self.dataset.X, self.dataset.y, test_size=test_frac)
        train_data = np.column_stack((X_train, y_train))
        test_data = np.column_stack((X_test, y_test))
        return Dataset(data=train_data), Dataset(data=test_data)

    def calc_generalization_error(self, program, model) -> float:
        self.observations, self.actual = self.test.get_observations(), self.test.get_actual()
        return self.evaluate(program, model)

    @override
    def cost(self, predictions: list) -> float:
        return super().cost(predictions)


class Conjunction(BooleanFunction):
    def __init__(self, n, negated_vars=False, k=1.3, dataset_type=DatasetType.COMPLETE):
        super().__init__(n_in=n, n_out=1, operator=lambda x, y: x & y,
                         negated_vars=negated_vars,
                         k=k,
                         dataset_type=dataset_type)


class ExclusiveDisjunction(BooleanFunction):
    def __init__(self, n, negated_vars=False, k=1.3, dataset_type=DatasetType.COMPLETE):
        super().__init__(n_in=n, n_out=1, operator=lambda x, y: x ^ y,
                         negated_vars=negated_vars,
                         k=k,
                         dataset_type=dataset_type)
