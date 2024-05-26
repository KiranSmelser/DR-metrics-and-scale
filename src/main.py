import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from zadu.measures import *
from datasets import *
from stress import *
from shepard import *
from viz import *


def main():
    datasets_dict = load_datasets()

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
                'random_shepard': shepard_corr[3],
                'mds_kruskal': kruskal_stress[2],
                'umap_kruskal': kruskal_stress[1],
                'tsne_kruskal': kruskal_stress[0],
                'random_kruskal': kruskal_stress[3]
            }
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


if __name__ == "__main__":
    main()
