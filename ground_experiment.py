import numpy as np
from sklearn import manifold
import umap.umap_ as umap
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler
from zadu.measures import *
from datasets import *
from stress import *


def main():
    expected_order = ['MDS', 'UMAP', 'Random']
    results_dict = {}
    results_order_dict = {'Run': [], 'Dataset': [],
                          'Initial Stress': [], '\"True\" Stress': [], 'Shepard': []}
    for i in range(10):
        datasets_dict = load_big_datasets()
        for dataset_name, (X, Y) in datasets_dict.items():
            if dataset_name not in results_dict:
                results_dict[dataset_name] = np.array([0, 0, 0])
            results = {}

            # Min-Max normalization
            X = MinMaxScaler().fit_transform(X)

            # # t-SNE
            # tsne = manifold.TSNE(n_components = 2, perplexity = 40, init = 'random')
            # fit_tsne = tsne.fit_transform(X)
            # init_stress = evaluate_scaling(X, fit_tsne, [1])[0]
            # min_stress = find_min_stress_exact(fit_tsne, X)[0]
            # shepard, _ = spearmanr(pdist(X), pdist(fit_tsne))
            # results['t-SNE'] = (init_stress, min_stress, shepard)

            # UMAP
            reducer = umap.UMAP(init='random')
            fit_umap = reducer.fit_transform(X)
            init_stress = evaluate_scaling(X, fit_umap, [1])[0]
            min_stress = find_min_stress_exact(fit_umap, X)[0]
            shepard, _ = spearmanr(pdist(X), pdist(fit_umap))
            results['UMAP'] = (init_stress, min_stress, shepard)

            # MDS
            mds = manifold.MDS(n_components=2, n_init=1, max_iter=120, n_jobs=2)
            fit_mds = mds.fit_transform(X)
            init_stress = evaluate_scaling(X, fit_mds, [1])[0]
            min_stress = find_min_stress_exact(fit_mds, X)[0]
            shepard, _ = spearmanr(pdist(X), pdist(fit_mds))
            results['MDS'] = (init_stress, min_stress, shepard)

            # Random Projection
            fit_random = np.random.uniform(0, 1, size=(X.shape[0], 2))
            init_stress = evaluate_scaling(X, fit_random, [1])[0]
            min_stress = find_min_stress_exact(fit_random, X)[0]
            shepard, _ = spearmanr(pdist(X), pdist(fit_random))
            results['Random'] = (init_stress, min_stress, shepard)

            # Check if order of both stress scores and Shepard correlations agree with expected order
            stress_init_order = sorted(
                results, key=lambda x: results[x][0])
            stress_min_order = sorted(
                results, key=lambda x: results[x][1])
            shepard_order = sorted(
                results, key=lambda x: results[x][2], reverse=True)
            counts = np.array([0, 0, 0])
            if stress_init_order == expected_order:
                counts[0] += 1
            if stress_min_order == expected_order:
                counts[1] += 1
            if shepard_order == expected_order:
                counts[2] += 1

            # Save the orderings for each trial
            results_order_dict['Dataset'].append(dataset_name)
            results_order_dict['Run'].append(i + 1)
            results_order_dict['Initial Stress'].append(
                ', '.join(stress_init_order))
            results_order_dict['\"True\" Stress'].append(
                ', '.join(stress_min_order))
            results_order_dict['Shepard'].append(', '.join(shepard_order))

            results_dict[dataset_name] += counts

    df = pd.DataFrame.from_dict(results_dict, orient='index', columns=[
                                'Initial Stress', '\"True\" Stress', 'Shepard'])
    df.to_csv('./results/ground experiment/experiment_results_umap.csv')

    df_order = pd.DataFrame(results_order_dict)
    df_order.to_csv(
        './results/ground experiment/experiment_orderings_umap.csv', index=False)


if __name__ == "__main__":
    main()
