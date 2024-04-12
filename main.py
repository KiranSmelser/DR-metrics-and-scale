import numpy as np
from sklearn import manifold
import umap.umap_ as umap
from sklearn import random_projection
from datasets import *
from stress import *
from embeddings import *
from shepard import *
from viz import *


def main():
    datasets_dict = load_datasets()
    range = 2
    scalars = np.linspace(0.0, range, 200)

    for dataset_name, (X, Y) in datasets_dict.items():
        # Normalize
        X /= np.max(X, axis=0)

        # t-SNE
        tsne = manifold.TSNE(n_components=2, perplexity=40,
                             init='pca', random_state=42)
        fit_tsne = tsne.fit_transform(X)
        tsne_stresses = evaluate_scaling(fit_tsne, X, scalars)

        # UMAP
        reducer = umap.UMAP(random_state=42)
        fit_umap = reducer.fit_transform(X)
        umap_stresses = evaluate_scaling(fit_umap, X, scalars)

        # MDS
        mds = manifold.MDS(n_components=2, n_init=1,
                           max_iter=120, n_jobs=2, random_state=42)
        fit_mds = mds.fit_transform(X)
        mds_stresses = evaluate_scaling(fit_mds, X, scalars)

        # Random Projection
        # transformer = random_projection.GaussianRandomProjection(
        #     n_components=2, random_state=42)
        # fit_random = transformer.fit_transform(X)
        
        #We actually wanted something a little less sophisticated
        fit_random = np.random.uniform(0,1, size=(X.shape[0], 2))
        random_stresses = evaluate_scaling(fit_random, X, scalars)

        # Save results
        results = {
            'tsne': (fit_tsne, tsne_stresses),
            'umap': (fit_umap, umap_stresses),
            'mds': (fit_mds, mds_stresses),
            'random': (fit_random, random_stresses)
        }

        # Plot embeddings
        plot_embeddings(results, dataset_name)

        # Plot Sheppard Diagrams
        shepard_scalars, shepard_corrs = shepard(
            X, results, range, dataset_name)

        # Orderings
        rankings = set(orderings(scalars, results))

        # Create summary plot
        summary_plot(scalars, shepard_scalars, results, dataset_name)

        for algo, (fit, stresses) in results.items():
            np.savetxt(f'{dataset_name}/{algo}_fit.txt', fit)
            np.savetxt(f'{dataset_name}/{algo}_stresses.txt', stresses)
        np.savetxt(f'{dataset_name}/shepard_scalars.txt', shepard_scalars)
        np.savetxt(f'{dataset_name}/shepard_corrs.txt', shepard_corrs)
        np.savetxt(f'{dataset_name}/rankings.txt', list(rankings), fmt='%s')
        np.savetxt(f'{dataset_name}/min_stresses.txt',
                   find_min_stress(results))
        np.savetxt(f'{dataset_name}/optimal_scalars.txt',
                   find_optimal_scalars(scalars, results))


if __name__ == "__main__":
    main()
