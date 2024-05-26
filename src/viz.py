"""Plots the low-dimensional embeddings of the dimensionality reduction techniques."""
import itertools
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.optimize import fsolve
from sklearn.preprocessing import MinMaxScaler

from datasets import *
from ranges import *


def orderings(scalars, techniques):
    """Function that determines all order changes among the Normalized Stress 
scores of the dimensionality reduction techniques within an "interesting" range."""
    rankings = []
    orderings = []
    for i in range(1, len(scalars)):
        current_order = np.argsort([techniques['tsne'][1][i], techniques['umap']
                                   [1][i], techniques['mds'][1][i], techniques['random'][1][i]])
        orderings.append(tuple(current_order))

        current_ranking = [list(techniques)[j] for j in current_order]
        rankings.append(tuple(current_ranking))
    return rankings


def plot_summary(techniques, dataset_name):
    """Function that plots the summary plot for all techniques."""
    plt.figure(figsize=(10, 8))

    # Store all curves
    curves = []

    for technique in techniques:
        stresses = [np.load(
            f'../results/{dataset_name}/stresses/stress_{j}_{technique}.npy') for j in range(10)]
        avg_stress = np.mean(stresses, axis=0)
        scalars = np.linspace(0.0, find_range(
            dataset_name), int(find_range(dataset_name)*100))

        plt.plot(scalars, avg_stress, label=technique)

        # Mark the minimum point on each curve
        min_index = np.argmin(avg_stress)
        plt.scatter(scalars[min_index], avg_stress[min_index], marker='x', label=f'{technique} min')

        # Store the curve
        curves.append((scalars, avg_stress))

    # Find and mark intersections
    intersection_label_added = False
    for (scalars1, avg_stress1), (scalars2, avg_stress2) in itertools.combinations(curves, 2):
        # Define function for the difference between the two curves
        func = lambda x : np.interp(x, scalars1, avg_stress1) - np.interp(x, scalars2, avg_stress2)
        # Find the intersection points (excluding 0)
        x_intersections = fsolve(func, scalars1[1:])
        # Mark the intersection points
        for x in x_intersections:
            if x >= 0 and x <= find_range(dataset_name):
                y = np.interp(x, scalars1, avg_stress1)

                if not intersection_label_added:
                    plt.scatter(x, y, color='gray', zorder=3, label='Intersection')
                    intersection_label_added = True
                else:
                    plt.scatter(x, y, color='gray', zorder=3)

    plt.yscale('log')
    plt.title(f'{dataset_name}')
    plt.ylabel('log(stress)')
    plt.legend(loc='best')
    plt.savefig(f'../results/{dataset_name}/summary.pdf')
    plt.close()


def plot_shepard(high, low, algo, dataset, run):
    """Function that plots the Shepard diagram for each technique."""
    if high.shape[0] > 200:
        idx = np.random.choice(high.shape[0], 200, replace=False)
        high = high[idx]
        low = low[idx]

    high = pdist(high)
    low = pdist(low)

    plt.figure(figsize=(8, 8))
    plt.scatter(high, low, s=.1)
    plt.xlabel('Low-dimensional distances')
    plt.ylabel('High-dimensional distances')
    plt.title(f'Shepard diagram of {dataset} for {algo}')
    plt.savefig(f'../results/{dataset}/shepard_plots/{algo}_shepard_{run}.pdf')
    plt.close()


def plot_embedding(low, algo, dataset, run):
    """Function that plots a single low-dimensional embedding."""
    df = pd.DataFrame(low, columns=[
                              'Component 1', 'Component 2'])
    plt.figure(figsize=(8, 8))
    plt.scatter(df['Component 1'], df['Component 2'])
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'Embedding of {dataset} with {algo}')
    plt.savefig(f'../results/{dataset}/embeddings/{algo}_embedding_{run}.pdf')
    plt.close()


if __name__ == "__main__":
    techniques = ['tsne', 'umap', 'mds', 'random']
    datasets_dict = load_datasets()

    for dataset_name, (X, Y) in tqdm.tqdm(datasets_dict.items()):
        # Min-Max normalization
        X = MinMaxScaler().fit_transform(X)

        plot_summary(techniques, dataset_name)

