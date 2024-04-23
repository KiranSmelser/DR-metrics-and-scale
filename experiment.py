import numpy as np
from sklearn import manifold
import umap.umap_ as umap
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler
from datasets import *
from stress import *


def experiment():
    expected_order = ['MDS', 'UMAP', 'Random']
    counts = [0, 0]
    for _ in range(10):
        datasets_dict = load_big_datasets()
        for dataset_name, (X, Y) in datasets_dict.items():
            results = {}

            # Min-Max normalization
            X = MinMaxScaler().fit_transform(X)

            # t-SNE
            # tsne = manifold.TSNE(n_components=2, perplexity=40,
            #                      init='pca', random_state=23)
            # fit_tsne = tsne.fit_transform(X)
            # tsne_stress = normalized_stress(pdist(X), pdist(fit_tsne))
            # tsne_shepard, _ = spearmanr(pdist(X), pdist(fit_tsne))
            # results['t-SNE'] = (tsne_stress, tsne_shepard)

            # UMAP
            reducer = umap.UMAP(random_state=23)
            fit_umap = reducer.fit_transform(X)
            umap_stress = normalized_stress(pdist(X), pdist(fit_umap))
            umap_shepard, _ = spearmanr(pdist(X), pdist(fit_umap))
            results['UMAP'] = (umap_stress, umap_shepard)

            # MDS
            mds = manifold.MDS(n_components=2, n_init=1,
                               max_iter=120, n_jobs=2, random_state=23)
            fit_mds = mds.fit_transform(X)
            mds_stress = normalized_stress(pdist(X), pdist(fit_mds))
            mds_shepard, _ = spearmanr(pdist(X), pdist(fit_mds))
            results['MDS'] = (mds_stress, mds_shepard)

            # Random Projection
            fit_random = np.random.uniform(0, 1, size=(X.shape[0], 2))
            random_stress = normalized_stress(pdist(X), pdist(fit_random))
            random_shepard, _ = spearmanr(pdist(X), pdist(fit_random))
            results['Random'] = (random_stress, random_shepard)

            # Check if order of both stress scores and Shepard correlations agree with expected order
            stress_order = sorted(
                results, key=lambda x: results[x][0])
            shepard_order = sorted(
                results, key=lambda x: results[x][1], reverse=True)
            if stress_order == expected_order:
                counts[0] += 1
            elif shepard_order == expected_order:
                counts[1] += 1
    np.savetxt(f'experiment_results.txt', counts)


experiment()
