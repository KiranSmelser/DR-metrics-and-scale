"""Plots the low-dimensional embeddings of the dimensionality reduction techniques."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
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


def plot_summary(X, techniques, dataset_name):
    """Function that plots the summary plot for all techniques."""
    plt.figure(figsize=(10, 8))

    for technique in techniques:
        stresses = [np.load(
            f'../results/{dataset_name}/stresses/stress_{j}_{technique}.npy') for j in range(10)]
        avg_stress = np.mean(stresses, axis=0)
        scalars = np.linspace(0.0, find_range(
            dataset_name), int(find_range(dataset_name)*100))

        for other_technique in techniques:
            if other_technique != technique:
                other_stresses = [np.load(
                    f'../results/{dataset_name}/stresses/stress_{j}_{other_technique}.npy') for j in range(10)]
                avg_other_stress = np.mean(other_stresses, axis=0)
                intersection = np.argwhere(np.diff(np.sign(np.array(avg_other_stress) - np.array(avg_stress)))).flatten()
                if len(intersection) > 1:
                    plt.axvline(x=scalars[intersection[1]], color='gray', linestyle='--')

        plt.plot(scalars, avg_stress, label=technique)

        # Mark the minimum point on each curve
        min_index = np.argmin(avg_stress)
        plt.scatter(scalars[min_index], avg_stress[min_index])

    plt.yscale("log")
    plt.title(f'{dataset_name} summary plot')
    plt.legend()
    plt.savefig(f'../results/{dataset_name}/summary.png')
    plt.close()


def plot_shepard(high, low, algo, dataset):
    """Function that plots the Shepard diagram for each technique."""
    if high.shape[0] > 1000:
        idx = np.random.choice(high.shape[0], 1000, replace=False)
        high = high[idx]
        low = low[idx]

    high = pdist(high)
    low = pdist(low)

    plt.figure(figsize=(8, 8))
    plt.scatter(high, low, s=.1)
    plt.xlabel('Low-dimensional distances')
    plt.ylabel('High-dimensional distances')
    plt.title(f'Shepard diagram for {algo}')
    plt.savefig(f'../results/{dataset}/{algo}_shepard.png')
    plt.close()


def plot_embedding(df, algo, subplot_position):
    """Function that plots a single low-dimensional embedding."""
    plt.subplot(2, 2, subplot_position)
    sns.scatterplot(data=df, x='Component 1', y='Component 2')
    plt.title(algo.capitalize())


if __name__ == "__main__":
    techniques = ['tsne', 'umap', 'mds', 'random']
    datasets_dict = load_datasets()

    for dataset_name, (X, Y) in datasets_dict.items():
        # Min-Max normalization
        X = MinMaxScaler().fit_transform(X)

        plt.figure(figsize=(15, 10))

        for i, technique in enumerate(techniques, 1):
            embeddings = [
                np.load(f'./data_embeddings/{dataset_name}_{j}_{technique}.npy') for j in range(10)]
            avg_embedding = np.mean(embeddings, axis=0)
            df = pd.DataFrame(avg_embedding, columns=[
                              'Component 1', 'Component 2'])

            plot_embedding(df, technique, i)
            plot_shepard(X, avg_embedding, technique, dataset_name)

        plot_summary(X, techniques, dataset_name)

        sns.set_palette("colorblind")
        plt.tight_layout()
        plt.savefig(f'../results/{dataset_name}/embeddings.png')
        plt.close()
