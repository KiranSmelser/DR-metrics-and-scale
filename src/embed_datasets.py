import os
import tqdm
import numpy as np
from pathlib import Path
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE, MDS
from umap import UMAP
import warnings

from datasets import *


class DimensionReducer():
    def __init__(self, X, labels):
        self.X = MinMaxScaler().fit_transform(X)
        self.labels = labels

        self.D = pairwise_distances(self.X)

    def compute_MDS(self):
        Y = MDS(dissimilarity="precomputed").fit_transform(self.D)
        return Y

    def compute_TSNE(self):
        Y = TSNE(metric='precomputed', init='random').fit_transform(self.D)
        return Y

    def compute_UMAP(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            Y = UMAP(metric='precomputed').fit_transform(self.D)
            return Y

    def compute_random(self):
        Y = np.random.uniform(0, 1, (self.X.shape[0], 2))
        return Y
    

def save_embeddings(data, name, i, folder):
    DR = DimensionReducer(*data)
    computations = ['MDS', 'TSNE', 'UMAP', 'random']
    methods = [DR.compute_MDS, DR.compute_TSNE, DR.compute_UMAP, DR.compute_random]

    for comp, method in zip(computations, methods):
        result = method()
        np.save(f"{folder}/{name}_{i}_{comp.lower()}.npy", result)
        if folder == 'big_data_embeddings':
            np.save(f"{folder}/{name}_{i}.npy", data[0])


if __name__ == "__main__":
    datasets = [('data_embeddings', load_datasets), ('big_data_embeddings', load_big_datasets)]
    
    for folder, loader in datasets:
        Path(folder).mkdir(exist_ok=True)
        for name, data in tqdm.tqdm(loader().items()):
            for i in range(10):
                save_embeddings(data, name, i, folder)
