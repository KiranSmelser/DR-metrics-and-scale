"""Module providing functions to calculate the Normalized Stress score, 
scale the Stress score within the "interesting" range, find the minimum Stress value, 
and find the scalar value that corresponds to the minimum Stress value for each 
dimensionality reduction technique."""

import numpy as np
from scipy.spatial.distance import pdist


def normalized_stress(D_high, D_low):
    """Function that calculates the Normalized Stress score."""
    num = (D_low - D_high)**2
    denom = D_high**2
    term = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)
    _sum = np.sum(term)
    return _sum


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
