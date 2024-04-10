import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

def shepard(X, tsne, umap, mds, max_scalar, dataset):
    scalars = []
    correlations = []

    projections = [tsne, umap, mds]
    algorithms = ['tsne', 'umap', 'mds']
    for i, projection in enumerate(projections):
        # Compute distances in high-dimensional and low-dimensional spaces
        dist_high = pdist(X)
        dist_low = pdist(projection)
        dist_high = dist_high.flatten()
        dist_low = dist_low.flatten()

        # Calculate the scalar for the axes
        scalar = dist_high.ptp() / dist_low.ptp() 
        scalars.append(scalar)
        dist_low *= scalar
        if (scalar > max_scalar):
            scalars = scalars[:-1]
            scalar = dist_low.ptp() / dist_high.ptp() 
            scalars.append(scalar)
            dist_high *= scalar

        # Compute the correlation between distances
        corr, _ = spearmanr(dist_high, dist_low)
        correlations.append(corr)

        plot_shepard(dist_high, dist_low, algorithms[i], dataset)
    
    return scalars, correlations

def plot_shepard(high, low, algo, dataset_name):
    plt.figure(figsize=(8, 8))
    plt.scatter(high, low, s=.1)
    plt.xlabel('Low-dimensional distances')
    plt.ylabel('High-dimensional distances')
    plt.title(f'Shepard diagram for {algo}')
    plt.savefig(f'{dataset_name}/{algo}_sheppard.png')
    plt.clf()