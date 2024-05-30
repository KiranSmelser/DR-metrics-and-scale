import tqdm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler

from zadu.measures import *
from datasets import *
from stress import *


def main():
    methods = ['umap', 'tsne']
    for method in methods:
        expected_order = ['mds', method, 'random']
        results_dict = {}
        results_order_dict = {'Run': [], 'Dataset': [],
                              'Initial Stress': [], '\"True\" Stress': [], 'Shepard': [], 'Kruskal': []}

        for i in tqdm.tqdm(range(1)):
            datasets = ['mnist', 'fmnist', 'spambase']

            for dataset_name in datasets:
                # Loda high-dimensional data from files
                X = np.load(f'./big_data_embeddings/{dataset_name}_{i}.npy')

                # Min-Max normalization
                X = MinMaxScaler().fit_transform(X)

                # Load the embeddings from the files
                fit_mds = np.load(
                    f'./big_data_embeddings/{dataset_name}_{i}_mds.npy')
                fit_random = np.load(
                    f'./big_data_embeddings/{dataset_name}_{i}_random.npy')

                if dataset_name not in results_dict:
                    results_dict[dataset_name] = np.array([0, 0, 0, 0])
                results = {}

                fit_method = np.load(
                    f'./big_data_embeddings/{dataset_name}_{i}_{method}.npy')

                init_stress = evaluate_scaling(X, fit_method, [1])[0]
                min_stress = find_min_stress_exact(fit_method, X)[0]
                shepard = spearman_rho.measure(X, fit_method)['spearman_rho']
                kruskal = compute_stress_kruskal(X, fit_method)
                results[method] = (init_stress, min_stress, shepard, kruskal)

                # MDS
                init_stress = evaluate_scaling(X, fit_mds, [1])[0]
                min_stress = find_min_stress_exact(fit_mds, X)[0]
                shepard = spearman_rho.measure(X, fit_mds)['spearman_rho']
                kruskal = compute_stress_kruskal(X, fit_mds)
                results['mds'] = (init_stress, min_stress, shepard, kruskal)

                # Random Projection
                init_stress = evaluate_scaling(X, fit_random, [1])[0]
                min_stress = find_min_stress_exact(fit_random, X)[0]
                shepard = spearman_rho.measure(X, fit_random)['spearman_rho']
                kruskal = compute_stress_kruskal(X, fit_random)
                results['random'] = (init_stress, min_stress, shepard, kruskal)

                # Check if order of both stress scores and Shepard correlations agree with expected order
                stress_init_order = sorted(
                    results, key=lambda x: results[x][0])
                stress_min_order = sorted(
                    results, key=lambda x: results[x][1])
                shepard_order = sorted(
                    results, key=lambda x: results[x][2], reverse=True)
                kruskal_order = sorted(
                    results, key=lambda x: results[x][3])
                counts = np.array([0, 0, 0, 0])
                if stress_init_order == expected_order:
                    counts[0] += 1
                if stress_min_order == expected_order:
                    counts[1] += 1
                if shepard_order == expected_order:
                    counts[2] += 1
                if kruskal_order == expected_order:
                    counts[3] += 1

                # Save the orderings for each trial
                results_order_dict['Dataset'].append(dataset_name)
                results_order_dict['Run'].append(i + 1)
                results_order_dict['Initial Stress'].append(
                    ', '.join(stress_init_order))
                results_order_dict['\"True\" Stress'].append(
                    ', '.join(stress_min_order))
                results_order_dict['Shepard'].append(', '.join(shepard_order))
                results_order_dict['Kruskal'].append(', '.join(kruskal_order))

                results_dict[dataset_name] += counts

        df = pd.DataFrame.from_dict(results_dict, orient='index', columns=[
                                    'Initial Stress', '\"True\" Stress', 'Shepard', 'Kruskal'])
        df.to_csv(
            f'../results/ground_experiment/experiment_results_{method}.csv')

        df_order = pd.DataFrame(results_order_dict)
        df_order.to_csv(
            f'../results/ground_experiment/experiment_orderings_{method}.csv', index=False)


if __name__ == "__main__":
    main()
