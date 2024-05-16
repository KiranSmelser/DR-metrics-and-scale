import numpy as np
import pandas as pd
from sklearn import manifold
import umap.umap_ as umap
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform

from datasets import *
from stress import *


def main():
    np.random.seed(23)
    datasets_dict = load_datasets()

    max_ranges = {}
    for dataset_name, (X, Y) in datasets_dict.items():

        # Min-Max normalization
        X = MinMaxScaler().fit_transform(X)

        # t-SNE
        tsne = manifold.TSNE(n_components=2, perplexity=40,
                             init='pca')
        fit_tsne = tsne.fit_transform(X)
        _, tsne_scalar = find_min_stress_exact(fit_tsne, X)

        # UMAP
        reducer = umap.UMAP()
        fit_umap = reducer.fit_transform(X)
        _, umap_scalar = find_min_stress_exact(fit_umap, X)

        # MDS
        mds = manifold.MDS(n_components=2, n_init=1,
                           max_iter=120, n_jobs=2)
        fit_mds = mds.fit_transform(X)
        _, mds_scalar = find_min_stress_exact(fit_mds, X)

        # Random Projection
        fit_random = np.random.uniform(0, 1, size=(X.shape[0], 2))
        _, random_scalar = find_min_stress_exact(fit_random, X)

        # Save results
        results = {
            'tsne': fit_tsne,
            'umap': fit_umap,
            'mds': fit_mds,
            'random': fit_random
        }

        # Calculate the intersection points
        intersection_points = {}
        for method_A, fit_A in results.items():
            for method_B, fit_B in results.items():
                if method_A != method_B:
                    intersection_points[(method_A, method_B)] = find_intersection(
                        squareform(pdist(X)), squareform(pdist(fit_A)), squareform(pdist(fit_B)))

        max_intersection = max(intersection_points.values())
        max_scalar = max([tsne_scalar, umap_scalar, mds_scalar, random_scalar])
        max_ranges[dataset_name] = max(max_intersection, max_scalar)
    df = pd.DataFrame(list(max_ranges.items()), columns=['dataset', 'max'])
    df.to_csv('../ranges.csv', index=False)


if __name__ == "__main__":
    main()
