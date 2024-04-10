import numpy as np
from scipy.spatial.distance import pdist

def normalized_stress(D_high, D_low):
    num = (D_low - D_high)**2
    denom = (D_high**2)
    term = np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)
    _sum = np.sum(term)
    return _sum

def evaluate_scaling(X_high, X_low, scalars):
    D_high = pdist(X_high)
    D_low = pdist(X_low)
    stresses = []
    for scalar in scalars:
        D_low_scaled = D_low * scalar
        stress = normalized_stress(D_high, D_low_scaled)
        stresses.append(stress)
    return stresses