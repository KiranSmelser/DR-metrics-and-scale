import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_embeddings(tsne, umap, mds, dataset):
    df_tsne = pd.DataFrame(tsne, columns=['Component 1', 'Component 2'])
    df_umap = pd.DataFrame(umap, columns=['Component 1', 'Component 2'])
    df_mds = pd.DataFrame(mds, columns=['Component 1', 'Component 2'])

    sns.set_palette("colorblind")

    # Create separate scatter plots for each technique
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df_tsne, x='Component 1', y='Component 2')
    plt.title('t-SNE')

    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df_umap, x='Component 1', y='Component 2')
    plt.title('UMAP')

    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df_mds, x='Component 1', y='Component 2')
    plt.title('MDS')

    plt.tight_layout()
    plt.savefig(f'{dataset}/embeddings.png')
    plt.clf()