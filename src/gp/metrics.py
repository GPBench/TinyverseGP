import numpy as np
from scipy.stats import pearsonr, spearmanr

def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def pearson_corr_loss(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    r, _ = pearsonr(y_true, y_pred)
    return 1 - r 

def spearman_corr_loss(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    r, _ = spearmanr(y_true, y_pred)
    return 1 - r

def cosine_sim_loss(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num = np.dot(y_true, y_pred)
    den = np.linalg.norm(y_true) * np.linalg.norm(y_pred)
    cos = num / den
    return 1 - cos
