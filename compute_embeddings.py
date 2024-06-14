import tqdm 
import warnings

import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS, TSNE 
from umap import UMAP

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
            Y = UMAP(metric='precomputed',init='random').fit_transform(self.D)
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
    import os
    if not os.path.isdir("embeddings"):
        os.mkdir('embeddings')

    num_iter = 10
    with tqdm.tqdm(total=len(os.listdir('datasets')) * num_iter) as pbar:

        for datasetStr in os.listdir("datasets"):
            X = np.load(f"datasets/{datasetStr}")
            DR = DimensionReducer(X,None)
            dname = datasetStr.replace(".npy", "")

            for i in range(num_iter):
                mds = DR.compute_MDS()
                np.save(f"embeddings/{dname}-MDS-{i}.npy",mds)

                tsne = DR.compute_TSNE()
                np.save(f"embeddings/{dname}-TSNE-{i}.npy",tsne)

                umap = DR.compute_UMAP()
                np.save(f"embeddings/{dname}-UMAP-{i}.npy",umap)

                random = DR.compute_random()
                np.save(f"embeddings/{dname}-RANDOM-{i}.npy",random)                                    
        
                pbar.update(1)