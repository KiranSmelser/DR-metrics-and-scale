# Import necessary libraries
import numpy as np 
import json 
import os
import tqdm

from metrics import Metrics


def compute_all_metrics():
    """
    Function to compute all metrics for each dataset in the 'datasets' directory.
    The results are saved in a JSON file.
    """
    results = dict()

    with tqdm.tqdm(total=len(os.listdir('datasets')) * 10 * 4) as pbar:

        for datasetStr in os.listdir("datasets"): 
            datasetName = datasetStr.replace(".npy", "")
            if "fashion_mnist" in datasetStr:
                datasetName = "fashion_mnist"   
            # Load and scale dataset      
            X = np.load(f"datasets/{datasetName}.npy")
            X *= 10
            
            print(f"Dataset: {datasetName}, size: {X.shape}")
            # Compute the metrics for each technique and dataset pair
            for i in range(10):
                datasetResults = dict()
                for alg in ["MDS", "TSNE", 'UMAP', "RANDOM"]:        
                    Y = np.load(f"embeddings/{datasetName}_{alg}_{i}.npy")

                    M = Metrics(X,Y)

                    # Compute and store each metric
                    datasetResults[f'{alg}_raw'] = M.compute_raw_stress()
                    datasetResults[f'{alg}_norm'] = M.compute_normalized_stress()
                    datasetResults[f'{alg}_scalenorm'] = M.compute_scale_normalized_stress()
                    datasetResults[f'{alg}_kruskal'] = M.compute_kruskal_stress()
                    datasetResults[f'{alg}_sheppard'] = M.compute_shepard_correlation()


                    pbar.update(1)

                datasetResults = {key: float(val) for key,val in datasetResults.items()}            
                results[f'{datasetName}_{i}'] = datasetResults    

            # Save the results to a JSON file
            with open("out10x.json", 'w') as fdata:
                json.dump(results,fdata,indent=4)            
                

if __name__ == "__main__":
    """
    Main function to compute all metrics for all dataset.
    """
    compute_all_metrics()