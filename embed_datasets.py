import numpy as np 
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE, MDS
from umap import UMAP

from datasets import load_datasets

class DimensionReducer():
    def __init__(self, X, labels):
        self.X = X 
        self.labels = labels

        self.D = pairwise_distances(self.X)

    def compute_MDS(self):
        Y = MDS(dissimilarity="precomputed").fit_transform(self.D)
        return Y

    def compute_TSNE(self):
        Y = TSNE(metric='precomputed').fit_transform(self.D)
        return Y

    def compute_UMAP(self):
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
        DR = DimensionReducer(*data)

        mds = DR.compute_MDS()
        np.save(f"data_embeddings/{name}_mds.npy", mds)

        tsne = DR.compute_TSNE()
        np.save(f"data_embeddings/{name}_tsne.npy", tsne)

        umap = DR.compute_UMAP()
        np.save(f"data_embeddings/{name}_umap.npy", umap)

        rand = DR.compute_random()
        np.save(f"data_embeddings/{name}_random.npy", rand)