import numpy as np 
from sklearn.metrics import pairwise_distances

import zadu

class Metrics():
    def __init__(self, X, Y):
        self.X = X 
        self.Y = Y 

        self.dX = pairwise_distances(X)
        self.dY = pairwise_distances(Y)

    def setY(self,Y):
        self.Y = Y 
        self.dY = pairwise_distances(Y)

    def compute_raw_stress(self):
        return np.sum(np.square(self.dX - self.dY)) / 2

    def compute_normalized_stress(self,alpha=1.0):        
        from zadu.measures import stress
        stressScore = stress.measure(self.X,alpha * self.Y,(self.dX, alpha * self.dY))
        return stressScore['stress']

    def compute_scale_normalized_stress(self,return_alpha=False):
        D_low_triu = self.dY[np.triu_indices(self.dY.shape[0], k=1)]
        D_high_triu = self.dX[np.triu_indices(self.dX.shape[0], k=1)]
        alpha = np.sum(D_low_triu * D_high_triu) / np.sum(np.square(D_low_triu))
        if return_alpha:
            return self.compute_normalized_stress(alpha), alpha
        return self.compute_normalized_stress(alpha)

    def compute_kruskal_stress(self):
        """
        Computes the non-metric stress between high dimensional distances D_high
        and low dimensional distances D_low. Invariant to scale of D_low.
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
        from zadu.measures import spearman_rho
        shepardCorr = spearman_rho.measure(self.X,self.Y,(self.dX,self.dY))
        return shepardCorr['spearman_rho']


if __name__ == "__main__":
    X = np.load('datasets/auto-mpg.npy')
    Y = np.load('embeddings/auto-mpg-TSNE-0.npy')

    M = Metrics(X,Y)
    
    scale_opt, alpha_opt = M.compute_scale_normalized_stress(return_alpha=True)

    rrange = np.linspace(0,100,2000)
    norm_stress_scores = list()
    for alpha in rrange:
        M.setY(alpha * Y)
        norm_stress_scores.append(M.compute_normalized_stress())

    import pylab as plt 
    plt.plot(rrange, norm_stress_scores)
    plt.scatter(alpha_opt, scale_opt)
    plt.show()

    

