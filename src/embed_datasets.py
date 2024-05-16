import numpy as np 
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE, MDS
from umap import UMAP
import warnings

from datasets import load_datasets

class DimensionReducer():
    def __init__(self, X, labels):
        self.X = MinMaxScaler().fit_transform(X)
        self.labels = labels

        self.D = pairwise_distances(self.X)

    def compute_MDS(self):
        Y = MDS(dissimilarity="precomputed").fit_transform(self.D)
        return Y

    def compute_TSNE(self):
        Y = TSNE(metric='precomputed',init='random').fit_transform(self.D)
        return Y

    def compute_UMAP(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            Y = UMAP(metric='precomputed').fit_transform(self.D)
            return Y
    
    def compute_random(self):
        Y = np.random.uniform(0,1,(self.X.shape[0], 2))
        return Y



if __name__ == "__main__":
    import tqdm as tqdm
    import os

    if not os.path.isdir("data_embeddings"):
        os.makedirs("data_embeddings")    

    for name, data in tqdm.tqdm(load_datasets().items()):
        for i in range(10):
            DR = DimensionReducer(*data)

            mds = DR.compute_MDS()
            np.save(f"data_embeddings/{name}_{i}_mds.npy", mds)

            tsne = DR.compute_TSNE()
            np.save(f"data_embeddings/{name}_{i}_tsne.npy", tsne)

            umap = DR.compute_UMAP()
            np.save(f"data_embeddings/{name}_{i}_umap.npy", umap)

            rand = DR.compute_random()
            np.save(f"data_embeddings/{name}_{i}_random.npy", rand)