import numpy as np
import pandas as pd
from sklearn import manifold
import umap.umap_ as umap
from sklearn.preprocessing import MinMaxScaler
from zadu.measures import *
from datasets import *
from stress import *
from shepard import *


def main():
    datasets_dict = load_datasets()
    num_runs = 10

    for dataset_name, (X, Y) in datasets_dict.items():
        # Min-Max normalization
        X = MinMaxScaler().fit_transform(X)

        all_results = []

        for i in range(num_runs):
            # t-SNE
            tsne = manifold.TSNE(
                n_components=2, perplexity=40, init='random')
            fit_tsne = tsne.fit_transform(X)

            # UMAP
            reducer = umap.UMAP(init='random')
            fit_umap = reducer.fit_transform(X)

            # MDS
            mds = manifold.MDS(n_components=2, n_init=1,
                               max_iter=120, n_jobs=2)
            fit_mds = mds.fit_transform(X)

            # Random Projection
            fit_random = np.random.uniform(0, 1, size=(X.shape[0], 2))

            # Save results
            results = {
                'tsne': (fit_tsne, 0),
                'umap': (fit_umap, 0),
                'mds': (fit_mds, 0),
                'random': (fit_random, 0)
            }

            # Shepard correlations
            shepard_corr = shepard(
                X, results, max, dataset_name)

            # Minimum stress and optimal scalars
            min_stress = {algo: find_min_stress_exact(
                fit, X) for algo, (fit, stresses) in results.items()}

            # Initial stress
            initial_stress = [evaluate_scaling(X, fit_tsne, [1])[0],
                              evaluate_scaling(X, fit_umap, [1])[0],
                              evaluate_scaling(X, fit_mds, [1])[0],
                              evaluate_scaling(X, fit_random, [1])[0]]

            run_results = {
                'Run': i+1,
                'mds_init': initial_stress[2],
                'umap_init': initial_stress[1],
                'tsne_init': initial_stress[0],
                'random_init': initial_stress[3],
                'mds_min': min_stress['mds'][0],
                'umap_min': min_stress['umap'][0],
                'tsne_min': min_stress['tsne'][0],
                'random_min': min_stress['random'][0],
                'mds_scalar': min_stress['mds'][1],
                'umap_scalar': min_stress['umap'][1],
                'tsne_scalar': min_stress['tsne'][1],
                'random_scalar': min_stress['random'][1],
                'mds_shepard': shepard_corr[2],
                'umap_shepard': shepard_corr[1],
                'tsne_shepard': shepard_corr[0],
                'random_shepard': shepard_corr[3]
            }
            all_results.append(run_results)

        # Save results
        df = pd.DataFrame(all_results)
        df.loc[len(df)] = df.mean()
        df.loc[df['Run'] == 5.5, 'Run'] = 'Average'
        df.to_csv(f'../results/experiment/{dataset_name}.csv', index=False)


if __name__ == "__main__":
    main()
