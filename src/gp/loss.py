"""
Provides various commonly used loss functions used as distance metrics
to calculate the fitness on candidate programs.
"""

import math
import numpy as np

# Loss functions commonly used to evaluate symbolic regression models


def absolute_error(x, y):
    return np.abs(np.subtract(x, y)).sum()

def mean_absolute_error(x, y):
    return np.abs(np.subtract(x, y)).mean()

def mean_squared_error(x, y):
    return np.square(np.subtract(x, y)).mean()


def root_mean_squared_error(x, y):
    return math.sqrt(mean_squared_error(x, y))


def linear_scaling_mse(x, y):
    if isinstance(x, list):
        x = np.array(x)
    if np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.all(x == x[0]):
        return np.mean((y - np.mean(y)) ** 2)
    q = np.hstack((np.reshape(x, (-1, 1)), np.ones((len(x), 1))))
    try:
        (a, b), mse, _, _ = np.linalg.lstsq(q, y, rcond=None)
        if mse.size > 0:
            return mse[0] / len(y)
    except Exception:
        pass
    return np.mean((y - np.mean(y)) ** 2)


def linear_scaling_coeff(x, y):
    if isinstance(x, list):
        x = np.array(x)
    if np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.all(x == x[0]):
        return 0, np.mean(y)
    q = np.hstack((np.reshape(x, (-1, 1)), np.ones((len(x), 1))))
    try:
        (a, b), mse, _, _ = np.linalg.lstsq(q, y, rcond=None)
        if mse.size > 0:
            return a, b
    except Exception:
        pass
    return 0, np.mean(y)


def euclidean_distance(x: list, y: list) -> float:
    """
    Calculate the Euclidean distance between two sets of values.

    :param x: Set of integers or real values
    :param y: Set of integers or real values
    :return: euclidean distance
    """
    if len(x) != len(y):
        raise ValueError("Dimensions do not match.")
    dist = 0.0
    for xi, yi in zip(x, y):
        dist += math.pow(xi - yi, 2)
    return math.sqrt(dist)


def absolute_distance(x: list, y: list) -> float:
    """
    Calculate the absolute distance between two sets of values.

    :param x: Set of integers or real values
    :param y: Set of integers or real values
    :return: absolute distance
    """
    if len(x) != len(y):
        raise ValueError("Dimensions do not match.")
    dist = 0.0
    for xi, yi in zip(x, y):
        dist += absolute_error(xi, yi)
    return dist


# Hamming distance is commonly used for logic synthesis


def hamming_distance(x: list, y: list) -> int:
    """
    Calculate the Hamming distance between two vectors.

    :param x: Set of (pseudo-boolean) values
    :param y: Set of (pseudo-boolean) values
    :return: hamming distance
    """
    if len(x) != len(y):
        raise ValueError("Dimensions do not match.")
    dist = 0
    for xi, yi in zip(x, y):
        if xi != yi:
            dist += 1
    return dist


def hamming_distance_bitwise(x: list, y: list, bin_length: int = 32) -> int:
    """
    Calculate the Hamming distance bitwise between two vectors of integer numbers.
    Commonly used as distance function for compressed truth tables.

    :param x: Set of integer values
    :param y: Set of integer values
    :param bin_length: Chunk size, which is the number of bits per chunk
    :return: hamming distance
    """
    dist = 0
    for xi, yi in zip(x, y):
        xi, yi = int(xi), int(yi)
        # Bitwise XOR the chunks to identify dissimilar bits
        cmp = xi ^ yi
        for i in range(bin_length):
            # Sum up the number of 1s then
            dist += cmp & 1
            cmp = cmp >> 1
    return dist
