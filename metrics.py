# Import necessary libraries
import numpy as np 
from sklearn.metrics import pairwise_distances

import zadu


class Metrics():
    """
    Class for computing various stress metrics between high-dimensional and low-dimensional data.
    """
    def __init__(self, X, Y):
        """
        Initialize the Metrics class with high-dimensional data X and low-dimensional data Y.
        Compute pairwise distances within X and Y.
        """
        self.X = X 
        self.Y = Y 

        self.dX = pairwise_distances(X)
        self.dY = pairwise_distances(Y)

    def setY(self,Y):
        """
        Update low-dimensional data Y and compute pairwise distances within Y.
        """
        self.Y = Y 
        self.dY = pairwise_distances(Y)

    def compute_raw_stress(self):
        """
        Compute raw stress between pairwise distances of X and Y.
        """
        return np.sum(np.square(self.dX - self.dY)) / 2

    def compute_normalized_stress(self,alpha=1.0):  
        """
        Compute normalized stress between X and alpha*Y using zadu's stress measure.
        """      
        from zadu.measures import stress
        stressScore = stress.measure(self.X,alpha * self.Y,(self.dX, alpha * self.dY))
        return stressScore['stress']

    def compute_scale_normalized_stress(self,return_alpha=False):
        """
        Compute scale-normalized stress between pairwise distances of X and Y.
        Optimal scaling factor alpha is computed as well.
        """
        D_low_triu = self.dY[np.triu_indices(self.dY.shape[0], k=1)]
        D_high_triu = self.dX[np.triu_indices(self.dX.shape[0], k=1)]
        alpha = np.sum(D_low_triu * D_high_triu) / np.sum(np.square(D_low_triu))
        if return_alpha:
            return self.compute_normalized_stress(alpha), alpha
        return self.compute_normalized_stress(alpha)

    def compute_kruskal_stress(self):
        """
        Compute Kruskal's non-metric stress between pairwise distances of X and Y. Invariant to scale of Y.
        """

        dij = self.dX[np.triu_indices(self.dX.shape[0], k=1)]
        xij = self.dY[np.triu_indices(self.dY.shape[0], k=1)]

        # Find the indices of dij that when reordered, would sort it. Apply to both arrays
        sorted_indices = np.argsort(dij)
        dij = dij[sorted_indices]
        xij = xij[sorted_indices]

        from sklearn.isotonic import IsotonicRegression
        hij = IsotonicRegression().fit(dij, xij).predict(dij)

        raw_stress = np.sum(np.square(xij - hij))
        norm_factor = np.sum(np.square(xij))

        kruskal_stress = np.sqrt(raw_stress / norm_factor)
        return kruskal_stress

    def compute_shepard_correlation(self):
        """
        Compute Shepard's correlation between pairwise distances of X and Y using zadu's spearman_rho measure.
        Invariant to scale of Y.
        """
        from zadu.measures import spearman_rho
        shepardCorr = spearman_rho.measure(self.X,self.Y,(self.dX,self.dY))
        return shepardCorr['spearman_rho']


if __name__ == "__main__":
    """
    Main function to compute and plot normalized stress for a range of scaling factors.
    """
    X = np.load('datasets/auto-mpg.npy')
    Y = np.load('embeddings/auto-mpg-TSNE-0.npy')

    M = Metrics(X,Y)
    
    scale_opt, alpha_opt = M.compute_scale_normalized_stress(return_alpha=True)

    # Compute normalized stress for a range of scaling factors
    rrange = np.linspace(0,100,2000)
    norm_stress_scores = list()
    for alpha in rrange:
        M.setY(alpha * Y)
        norm_stress_scores.append(M.compute_normalized_stress())

    # Plot normalized stress scores and highlight the optimal scaling factor
    import pylab as plt 
    plt.plot(rrange, norm_stress_scores)
    plt.scatter(alpha_opt, scale_opt)
    plt.show()

    

