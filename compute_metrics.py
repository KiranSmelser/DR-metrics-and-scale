import numpy as np 
import json 
import os
import tqdm

from metrics import Metrics


def compute_all_metrics():
    results = dict()
    for datasetStr in tqdm.tqdm(os.listdir("embeddings")):
        datasetName = datasetStr.replace(".npy", "")
        X = np.load(f"dataset/{datasetStr}")
        
        for i in range(10):
            datasetResults = dict()
            for alg in ["MDS", "TSNE", 'UMAP', "RANDOM"]:        
                Y = np.load(f"embeddings/{datasetName}-{alg}-{i}.npy")

                M = Metrics(X,Y)

                datasetResults[f'{alg}_raw'] = M.compute_raw_stress()
                datasetResults[f'{alg}_norm'] = M.compute_normalized_stress()
                datasetResults[f'{alg}_scalenorm'] = M.compute_scale_normalized_stress()
                datasetResults[f'{alg}_kruskal'] = M.compute_kruskal_stress()
                datasetResults[f'{alg}_sheppard'] = M.compute_shepard_correlation()
           
            results[f'{datasetName}_{i}'] = datasetResults    

    with open("out.json", 'w') as fdata:
        json.dump(results,fdata,indent=4)            
                


if __name__ == "__main__":
    compute_all_metrics()