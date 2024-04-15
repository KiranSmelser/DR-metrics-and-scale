import numpy as np
from sklearn import manifold
import umap.umap_ as umap
from sklearn import random_projection
from sklearn.preprocessing import MinMaxScaler
from datasets import *
from stress import *
from embeddings import *
from shepard import *
from viz import *


def main():
    datasets_dict = load_datasets()

    for dataset_name, (X, Y) in datasets_dict.items():
        range = find_range(dataset_name)
        scalars = np.linspace(0.0, range, int(range*100))


        # Min-Max normalization
        X = MinMaxScaler().fit_transform(X)

        # t-SNE
        tsne = manifold.TSNE(n_components=2, perplexity=40,
                             init='pca', random_state=23)
        fit_tsne = tsne.fit_transform(X)
        tsne_stresses = evaluate_scaling(fit_tsne, X, scalars)

        # UMAP
        reducer = umap.UMAP(random_state=23)
        fit_umap = reducer.fit_transform(X)
        umap_stresses = evaluate_scaling(fit_umap, X, scalars)

        # MDS
        mds = manifold.MDS(n_components=2, n_init=1,
                           max_iter=120, n_jobs=2, random_state=23)
        fit_mds = mds.fit_transform(X)
        mds_stresses = evaluate_scaling(fit_mds, X, scalars)

        # Random Projection
        fit_random = np.random.uniform(0, 1, size=(X.shape[0], 2))
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
        shepard_scalars_stresses = [results['tsne'][1][np.argmin(np.abs(scalars - shepard_scalars[0]))],
                                    results['umap'][1][np.argmin(
                                        np.abs(scalars - shepard_scalars[1]))],
                                    results['mds'][1][np.argmin(
                                        np.abs(scalars - shepard_scalars[2]))],
                                    results['random'][1][np.argmin(
                                        np.abs(scalars - shepard_scalars[3]))]]

        # Orderings
        rankings = set(orderings(scalars, results))

        # Create summary plot
        summary_plot(scalars, shepard_scalars, results, dataset_name)

        for algo, (fit, stresses) in results.items():
            np.savetxt(f'{dataset_name}/{algo}_fit.txt', fit)
            np.savetxt(f'{dataset_name}/{algo}_stresses.txt', stresses)
        np.savetxt(f'{dataset_name}/shepard_scalars.txt',
                   shepard_scalars_stresses)
        np.savetxt(f'{dataset_name}/shepard_corrs.txt', shepard_corrs)
        np.savetxt(f'{dataset_name}/rankings.txt', list(rankings), fmt='%s')
        np.savetxt(f'{dataset_name}/min_stresses_and_scalars.txt',
                   np.array([find_min_stress_exact(fit, X) for algo, (fit, stresses) in results.items()]))
        np.savetxt(f'{dataset_name}/min_stresses.txt', find_min_stress(results))
        np.savetxt(f'{dataset_name}/optimal_scalars.txt',
                   find_optimal_scalars(scalars, results))


if __name__ == "__main__":
    main()
