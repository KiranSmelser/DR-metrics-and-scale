"""Module providing functions that determine all order changes among the 
Normalized Stress scores of the dimensionality reduction techniques within 
the "interesting" range and create a summary plot used to compared the different techniques."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def summary_plot(scalars, sheppard, techniques, dataset):
    """Function that creates a summary plot used to compared the different 
dimensionality reduction techniques."""
    tsne_data = pd.DataFrame({
        'Scalar': scalars,
        'Normalized Stress': techniques['tsne'][1],
        'Method': 't-SNE'
    })

    umap_data = pd.DataFrame({
        'Scalar': scalars,
        'Normalized Stress': techniques['umap'][1],
        'Method': 'UMAP'
    })

    mds_data = pd.DataFrame({
        'Scalar': scalars,
        'Normalized Stress': techniques['mds'][1],
        'Method': 'MDS'
    })

    random_data = pd.DataFrame({
        'Scalar': scalars,
        'Normalized Stress': techniques['random'][1],
        'Method': 'Random'
    })

    data = pd.concat([tsne_data, umap_data, mds_data, random_data])

    plt.figure(figsize=(20, 12))
    sns.set(style="whitegrid")
    sns.lineplot(x='Scalar', y='Normalized Stress',
                 hue='Method', palette="colorblind", data=data)
    plt.xlabel('Scalar')
    plt.ylabel('Normalized Stress')

    # Points where the Sheppard Goodness correlation coefficients fall
    tsne_point, = plt.plot(sheppard[0], techniques['tsne'][1][np.argmin(
        np.abs(scalars - sheppard[0]))], 'x', color='blue', markersize=15)
    umap_point, = plt.plot(sheppard[1], techniques['umap'][1][np.argmin(
        np.abs(scalars - sheppard[1]))], 'x', color='orange', markersize=15)
    mds_point, = plt.plot(sheppard[2], techniques['mds'][1][np.argmin(
        np.abs(scalars - sheppard[2]))], 'x', color='green', markersize=15)
    random_point, = plt.plot(sheppard[2], techniques['random'][1][np.argmin(
        np.abs(scalars - sheppard[2]))], 'x', color='red', markersize=15)

    # Initial order
    initial_scalar_index = np.argmin(np.abs(scalars - 1))
    initial_order = np.argsort(
        [techniques['tsne'][1][initial_scalar_index], techniques['umap'][1][initial_scalar_index],
         techniques['mds'][1][initial_scalar_index], techniques['random'][1][initial_scalar_index]])
    labels = ['t-SNE', 'UMAP', 'MDS', 'Random']
    initial_ranking = [labels[i] for i in initial_order]
    plt.axvline(x=1, color='grey', linestyle='--')
    ranking_str = "\n" + \
        "\n".join(
            [f"{i+1}. {initial_ranking[i]}" for i in range(len(initial_ranking))])
    plt.annotate(f'Initial Ranking: {ranking_str}', xy=(1, 0.5), xycoords='axes fraction',
                 xytext=(1/max(scalars)-.015, .7), textcoords='axes fraction',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Check for changes in the order
    annotations = 1
    previous_order = initial_order
    for i in range(1, len(scalars)):
        current_order = np.argsort(
            [techniques['tsne'][1][i], techniques['umap'][1][i],
             techniques['mds'][1][i], techniques['random'][1][i]])
        if not np.array_equal(previous_order, current_order):
            current_ranking = [labels[j] for j in current_order]
            ranking_str = "\n" + \
                "\n".join(
                    [f"{j+1}. {current_ranking[j]}" for j in range(len(current_ranking))])
            plt.axvline(x=scalars[i], color='grey', linestyle='--')
            plt.annotate(f'Ranking when scaled by {scalars[i]:.2f}: {ranking_str}',
                         xy=(scalars[i], np.mean([techniques['tsne'][0][i],
                                                  techniques['umap'][0][i],
                                                  techniques['mds'][0][i],
                                                  techniques['random'][0][i]])),
                         xytext=(scalars[i],
                                 np.mean([techniques['tsne'][1][i],
                                          techniques['umap'][1][i],
                                          techniques['mds'][1][i],
                                          techniques['random'][1][i]])
                                 * annotations),
                         textcoords=('data'),
                         bbox=dict(facecolor='white',
                                   edgecolor='black',
                                   boxstyle='round,pad=0.5'))
            previous_order = current_order
            annotations += 1

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([tsne_point, umap_point, mds_point, random_point])
    labels.extend(['t-SNE Sheppard Scalar',
                  'UMAP Sheppard Scalar', 'MDS Sheppard Scalar', 'Random Sheppard Scalar'])
    plt.legend(handles=handles, labels=labels)
    plt.savefig(f'{dataset}/summary_plot.png')
    plt.clf()
