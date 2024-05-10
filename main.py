import numpy as np
from sklearn import manifold
import umap.umap_ as umap
from sklearn.preprocessing import MinMaxScaler
from zadu.measures import *
from datasets import *
from stress import *
from embeddings import *
from shepard import *
from viz import *


def main():
    np.random.seed(23)
    datasets_dict = load_datasets()

    for dataset_name, (X, Y) in datasets_dict.items():
        # range = find_range(dataset_name)
        range = 1
        scalars = np.linspace(0.0, range, int(range*100))

        # Min-Max normalization
        X = MinMaxScaler().fit_transform(X)

        # t-SNE
        tsne = manifold.TSNE(n_components=2, perplexity=40,
                             init='pca')
        fit_tsne = tsne.fit_transform(X)
        tsne_stresses = evaluate_scaling(X, fit_tsne, scalars)

        # UMAP
        reducer = umap.UMAP()
        fit_umap = reducer.fit_transform(X)
        umap_stresses = evaluate_scaling(X, fit_umap, scalars)

        # MDS
        mds = manifold.MDS(n_components=2, n_init=1,
                           max_iter=120, n_jobs=2)
        fit_mds = mds.fit_transform(X)
        mds_stresses = evaluate_scaling(X, fit_mds, scalars)

        # Random Projection
        fit_random = np.random.uniform(0, 1, size=(X.shape[0], 2))
        random_stresses = evaluate_scaling(X, fit_random, scalars)

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
        shepard(X, results, range, dataset_name)

        # Orderings
        rankings = set(orderings(scalars, results))

        # Create summary plot
        # summary_plot(scalars, shepard_scalars, results, dataset_name)

        # Save projections and stresses
        with pd.ExcelWriter(f'{dataset_name}/projections.xlsx') as writer:
            for algo, (fit, stresses) in results.items():
                pd.DataFrame(fit).to_excel(writer, sheet_name=f'{algo}_fit', header=False, index=False)

        with pd.ExcelWriter(f'{dataset_name}/stresses.xlsx') as writer:
            for algo, (fit, stresses) in results.items():
                pd.DataFrame(stresses).to_excel(writer, sheet_name=f'{algo}_stresses', header=False, index=False)

        # Save rankings
        rankings_df = pd.DataFrame(rankings).transpose()
        rankings_df.columns = [f'Ranking {i+1}' for i, _ in enumerate(rankings_df.columns)]
        rankings_df.to_csv(f'{dataset_name}/rankings.csv',
                           header=True, index=False)


if __name__ == "__main__":
    main()
