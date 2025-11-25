import numpy as np
from gp import metrics


def test_mse_zero_on_identical():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics.mse(y, y) == 0.0


def test_pearson_corr_loss_identical():
    y = np.array([1.0, 2.0, 3.0])
    loss = metrics.pearson_corr_loss(y, y)
    # identical vectors → correlation ≈ 1 → loss ≈ 0
    assert loss < 1e-6


def test_spearman_corr_loss_identical():
    y = np.array([1.0, 2.0, 3.0])
    loss = metrics.spearman_corr_loss(y, y)
    assert loss < 1e-6


def test_cosine_sim_loss_identical():
    y = np.array([1.0, 2.0, 3.0])
    loss = metrics.cosine_sim_loss(y, y)
    assert loss < 1e-6
