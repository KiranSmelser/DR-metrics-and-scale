import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from zadu.measures import *
from datasets import *
from stress import *


def calculate_metrics(hd, ld):
    # Compute the initial and minimum stress
    init_stress = evaluate_scaling(hd, ld, [1])[0]
    min_stress = find_min_stress_exact(ld, hd)[0]

    # Compute the Shepard correlation between distances
    shepard = spearman_rho.measure(hd, ld)['spearman_rho']

    # Compute the Kruskal stress
    kruskal = compute_stress_kruskal(hd, ld)

    # Compute the trustworthiness and continuity
    tmp = trustworthiness_continuity.measure(hd, ld)
    trustworthiness = tmp['trustworthiness']
    continuity = tmp['continuity']

    # Compute the MRREs
    tmp = mean_relative_rank_error.measure(hd, ld)
    false = tmp['mrre_false']
    missing = tmp['mrre_missing']

    # Compute the neighborhood hits
    hit = neighborhood_hit.measure(
        hd, ld)['neighborhood_hit']

    # Compute the Kullback-Leibler Divergence
    divergence = kl_divergence.measure(
        hd, ld)['kl_divergence']

    # Compute the Pearson correlation between distances
    corr = pearson_r.measure(hd, ld)['pearson_r']

    return (init_stress, min_stress, kruskal, false, missing, divergence, shepard, trustworthiness, continuity, hit, corr)


def main():
    methods = ['umap', 'tsne']

    # Define the order of the results and their corresponding names
    orders = ['Initial Stress', '\"True\" Stress', 'Kruskal', 'MMRE False', 'MMRE Missing',
              'KL Divergence', 'Shepard', 'Trustworthiness', 'Continuity', 'Neighborhood Hit',
              'Pearson']
    reverse_orders = ['Shepard', 'Trustworthiness',
                      'Continuity', 'Neighborhood Hit', 'Pearson']
    for method in methods:
        expected_order = ['mds', method, 'random']
        results_dict = {}

        results_order_dict = {'Run': [], 'Dataset': []}
        for order in orders:
            results_order_dict[order] = []

        for i in tqdm.tqdm(range(10)):
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
                    results_dict[dataset_name] = np.zeros(len(orders))
                results = {}

                fit_method = np.load(
                    f'./big_data_embeddings/{dataset_name}_{i}_{method}.npy')

                results[method] = calculate_metrics(X, fit_method)

                # MDS
                results['mds'] = calculate_metrics(X, fit_mds)

                # Random Projection
                results['random'] = calculate_metrics(X, fit_random)

                # Initialize counts
                counts = np.zeros(len(orders))

                # Check if order of quality metrics agree with expected order
                for j, order in enumerate(orders):
                    sorted_order = sorted(
                        results, key=lambda x: results[x][j], reverse=order in reverse_orders)
                    if sorted_order == expected_order:
                        counts[j] += 1
                    results_order_dict[order].append(', '.join(sorted_order))

                # Save the orderings for each trial
                results_order_dict['Dataset'].append(dataset_name)
                results_order_dict['Run'].append(i + 1)

                results_dict[dataset_name] += counts

        df = pd.DataFrame.from_dict(
            results_dict, orient='index', columns=orders)
        df.to_csv(
            f'../results/ground_experiment/experiment_results_{method}.csv')

        df_order = pd.DataFrame(results_order_dict)
        df_order.to_csv(
            f'../results/ground_experiment/experiment_orderings_{method}.csv', index=False)


if __name__ == "__main__":
    main()
