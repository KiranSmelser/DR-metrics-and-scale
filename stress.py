"""Module providing functions to calculate the Normalized Stress score, 
scale the Stress score within the "interesting" range, find the minimum Stress value, 
and find the scalar value that corresponds to the minimum Stress value for each 
dimensionality reduction technique."""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def normalized_stress(D_high, D_low):
    """Function that calculates the Normalized Stress score."""
    num = (D_low - D_high)**2
    denom = D_high**2
    term = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)
    _sum = np.sum(term)
    return _sum / 2


def evaluate_scaling(X_high, X_low, scalars):
    """Function that scales the Stress score within an "interesting" range."""
    D_high = pdist(X_high)
    D_low = pdist(X_low)
    stresses = []
    for scalar in scalars:
        D_low_scaled = D_low * scalar
        stress = normalized_stress(D_high, D_low_scaled)
        stresses.append(stress)
    return stresses


def find_min_stress(techniques):
    """Function that finds the minimum Stress score."""
    return [min(techniques['tsne'][1]), min(techniques['umap'][1]),
            min(techniques['mds'][1]), min(techniques['random'][1])]


def find_optimal_scalars(scalars, techniques):
    """Function that finds the scalar value which corresponds to the minimum Stress value."""
    return [scalars[techniques['tsne'][1].index(min(techniques['tsne'][1]))],
            scalars[techniques['umap'][1].index(min(techniques['umap'][1]))],
            scalars[techniques['mds'][1].index(min(techniques['mds'][1]))],
            scalars[techniques['random'][1].index(min(techniques['random'][1]))]]


def find_min_stress_exact(X, Y):
    """
    Given high dimensional data Y, low dimensional data X, 
    calculates the minimum value of the stress curve exactly.
    """
    D_low = squareform(pdist(X))
    D_high = squareform(pdist(Y))

    alpha = find_optimal_scalars_exact(D_low, D_high)

    return (normalized_stress(D_high, alpha * D_low), alpha)


def find_optimal_scalars_exact(D_low, D_high):
    """
    Given two distance matrices D_low, D_high, computes 
    the optimal scalar value to multiply D_low by so that 
    the normalized stress between them is minimum.
    """

    D_low_triu = D_low[np.triu_indices(D_low.shape[0], k=1)]
    D_high_triu = D_high[np.triu_indices(D_high.shape[0], k=1)]
    return np.sum(D_low_triu / D_high_triu) / np.sum(np.square(D_low_triu / D_high_triu))
