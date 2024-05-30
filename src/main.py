import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from zadu.measures import *
from datasets import *
from stress import *
from other_measures import *
from viz import *


def compare_rankings(ranking_df):
    metrics = ranking_df.columns
    comparison_results = {}

    for i in range(len(metrics)):
        for j in range(i+1, len(metrics)):
            metric1 = metrics[i]
            metric2 = metrics[j]
            # Check if the rankings are the same
            same_ranking = (ranking_df[metric1] == ranking_df[metric2]).all()
            comparison_results[(metric1, metric2)] = int(same_ranking)

    return comparison_results


def main():
    datasets_dict = load_datasets()
    all_datasets_results = {}

    for dataset_name, (X, Y) in tqdm.tqdm(datasets_dict.items()):
        # Min-Max normalization
        X = MinMaxScaler().fit_transform(X)

        rrange = find_range(dataset_name)
        scalars = np.linspace(0.0, rrange, int(rrange*100))
        all_results = []

        for i in range(10):
            # Load the embeddings from the files
            fit_mds = np.load(f'./data_embeddings/{dataset_name}_{i}_mds.npy')
            fit_umap = np.load(
                f'./data_embeddings/{dataset_name}_{i}_umap.npy')
            fit_tsne = np.load(
                f'./data_embeddings/{dataset_name}_{i}_tsne.npy')
            fit_random = np.load(
                f'./data_embeddings/{dataset_name}_{i}_random.npy')

            # Calculate stress values
            tsne_stresses = evaluate_scaling(X, fit_tsne, scalars)
            umap_stresses = evaluate_scaling(X, fit_umap, scalars)
            mds_stresses = evaluate_scaling(X, fit_mds, scalars)
            random_stresses = evaluate_scaling(X, fit_random, scalars)

            # Save results
            results = {
                'tsne': (fit_tsne, tsne_stresses),
                'umap': (fit_umap, umap_stresses),
                'mds': (fit_mds, mds_stresses),
                'random': (fit_random, random_stresses)
            }

            # Shepard correlations
            shepard_corr = shepard(X, results)

            # Trustworthiness & Continuity
            trustworthiness, continuity = trustworthiness_and_continuity(
                X, results)

            # MRRE
            mrre_false, mrre_missing = mrre(X, results)

            # Neighborhood Hit
            hits = neighbor_hit(X, results)

            # Steadiness & Cohesiveness
            steadiness, cohesiveness = steadiness_and_cohesiveness(
                X, results)

            # Kullback-Leibler Divergence
            kl_divergences = kl(X, results)

            # Pearson correlations
            pearson_corrs = pearson_corr(X, results)

            # Minimum stress and optimal scalars
            min_stress = {algo: find_min_stress_exact(
                fit, X) for algo, (fit, stresses) in results.items()}

            # Initial stress
            initial_stress = [evaluate_scaling(X, fit_tsne, [1])[0],
                              evaluate_scaling(X, fit_umap, [1])[0],
                              evaluate_scaling(X, fit_mds, [1])[0],
                              evaluate_scaling(X, fit_random, [1])[0]]

            # Kruskal stress
            kruskal_stress = [compute_stress_kruskal(X, fit_tsne),
                              compute_stress_kruskal(X, fit_umap),
                              compute_stress_kruskal(X, fit_mds),
                              compute_stress_kruskal(X, fit_random)]

            methods = ['tsne', 'umap', 'mds', 'random']
            metrics = ['init', 'min', 'scalar', 'shepard', 'kruskal', 'trustworthiness', 'continuity', 'mrre_false',
                       'mrre_missing', 'neighborhood_hit', 'steadiness', 'cohesiveness', 'kl_divergence', 'pearson']
            data_sources = [initial_stress, min_stress, min_stress, shepard_corr, kruskal_stress, trustworthiness,
                            continuity, mrre_false, mrre_missing, hits, steadiness, cohesiveness, kl_divergences, pearson_corrs]

            run_results = {'Run': i+1}
            for method, index in zip(methods, range(4)):
                for metric, data_source in zip(metrics, data_sources):
                    if metric in ['min', 'scalar']:
                        run_results[f'{method}_{metric}'] = data_source[method][metrics.index(metric)-1]
                    else:
                        run_results[f'{method}_{metric}'] = data_source[index]

            all_results.append(run_results)

            for algo, (fit, stresses) in results.items():
                np.save(
                    f"../results/{dataset_name}/stresses/stress_{i}_{algo}.npy", stresses)

            # Plots
            for algo, (fit, stresses) in results.items():
                plot_shepard(X, fit, algo, dataset_name, i)
                plot_embedding(fit, algo, dataset_name, i)

        # Save results
        df = pd.DataFrame(all_results)
        df.loc[len(df)] = df.mean()
        df.loc[df['Run'] == 5.5, 'Run'] = 'Average'
        df.to_csv(f'../results/experiment/{dataset_name}.csv', index=False)

        avg_row = df.loc[df['Run'] == 'Average']
        ranking_df = pd.DataFrame(index=methods)

        metrics.remove('scalar')
        for metric in metrics:
            metric_cols = [f'{method}_{metric}' for method in methods]
            metric_avgs = avg_row[metric_cols].values[0]
            if metric in ['shepard', 'trustworthiness', 'continuity', 'neighborhood_hit', 'pearson']:
                metric_ranks = pd.Series(
                    metric_avgs, index=methods).rank(ascending=False)
            elif metric in ['init', 'min', 'kruskal', 'mrre_false', 'mrre_missing', 'steadiness', 'cohesiveness', 'kl_divergence']:
                metric_ranks = pd.Series(
                    metric_avgs, index=methods).rank(ascending=True)
            ranking_df[metric] = metric_ranks

        ranking_df.to_csv(
            f'../results/experiment/{dataset_name}_rankings.csv', index=True)
        
        # Compare rankings
        comparison_results = compare_rankings(ranking_df)
        all_datasets_results[dataset_name] = comparison_results

    # Calculate agreement percentages
    metrics_pairs = list(all_datasets_results.values())[0].keys()
    agreement_percentages = {}

    for pair in metrics_pairs:
        agreement_counts = [results[pair]
                            for results in all_datasets_results.values()]
        agreement_percentages[pair] = sum(
            agreement_counts) / len(agreement_counts) * 100

    # Export to .csv
    df = pd.DataFrame.from_dict(agreement_percentages, orient='index', columns=[
                                'Agreement Percentage']).T
    df.to_csv('../results/experiment/agreement_percentages.csv')


if __name__ == "__main__":
    main()
