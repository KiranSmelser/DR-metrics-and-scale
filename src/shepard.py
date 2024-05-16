"""Module providing functions to calculate the Shepard Goodness score, plot 
the Shepard diagram, and calculate the scalar which makes the area of the 
Shepard diagram square for each dimensionality reduction technique."""

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


def shepard(X, techniques):
    """Function that calculates the Shepard Goodness score and the scalar which makes the 
area of the Shepard diagram square."""
    correlations = []

    projections = [techniques['tsne'][0], techniques['umap'][0],
                   techniques['mds'][0], techniques['random'][0]]
    for projection in projections:
        # Compute distances in high-dimensional and low-dimensional spaces
        dist_high = pdist(X)
        dist_low = pdist(projection)
        dist_high = dist_high.flatten()
        dist_low = dist_low.flatten()

        # Compute the correlation between distances
        corr, _ = spearmanr(dist_high, dist_low)
        correlations.append(corr)

    return correlations
