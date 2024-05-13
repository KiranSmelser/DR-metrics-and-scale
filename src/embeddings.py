"""Module providing functions that plot the low-dimensional 
embeddings of the dimensionality reduction techniques."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_embeddings(techniques, dataset):
    """Function that plots the low-dimensional embeddings of the 
    dimensionality reduction techniques."""
    df_tsne = pd.DataFrame(techniques['tsne'][0], columns=[
                           'Component 1', 'Component 2'])
    df_umap = pd.DataFrame(techniques['umap'][0], columns=[
                           'Component 1', 'Component 2'])
    df_mds = pd.DataFrame(techniques['mds'][0], columns=[
                          'Component 1', 'Component 2'])
    df_random = pd.DataFrame(techniques['random'][0], columns=[
                             'Component 1', 'Component 2'])

    sns.set_palette("colorblind")

    # Create separate scatter plots for each technique
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df_tsne, x='Component 1', y='Component 2')
    plt.title('t-SNE')

    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df_umap, x='Component 1', y='Component 2')
    plt.title('UMAP')

    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df_mds, x='Component 1', y='Component 2')
    plt.title('MDS')

    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df_random, x='Component 1', y='Component 2')
    plt.title('Random')

    plt.tight_layout()
    plt.savefig(f'../results/{dataset}/embeddings.png')
    plt.clf()
