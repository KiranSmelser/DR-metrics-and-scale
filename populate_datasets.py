import tqdm 
from sklearn import datasets
import seaborn as sns
import urllib.request

import os
if not os.path.isdir("datasets"):
    os.mkdir("datasets")

def loadEspadatoDatasets():

    #Grab the website with DR dataset links
    datasetHtml = str(urllib.request.urlopen("https://mespadoto.github.io/proj-quant-eval/post/datasets").read())

    #I looked over the html, saw that this would get the block of links for each dataset
    datasetList = datasetHtml.split("</tr>")[1:-1]

    for dataset in tqdm.tqdm(datasetList):
        """
        I want to remove header from the start and tail from the end of the third element of the <tr> list
        This gives me the dataset name; Luckily all datasets are stored on the Espadato website and have 
        the same directory.
        """
        header = "<td><a href=\"../../data"
        tail = "\">X.npy</a>"

        qstr = dataset.split("</td>")[3]
        qstr = qstr.replace(header,"https://mespadoto.github.io/proj-quant-eval/data")
        qstr = qstr.replace(tail, "").replace("\\n", "")

        name = qstr.replace("https://mespadoto.github.io/proj-quant-eval/data/", "").replace("/X.npy", "")

        data = urllib.request.urlopen(qstr)

        #If we write the raw binary to file, Numpy will know how to read it later.
        with open(f'datasets/{name}.npy', 'wb') as fdata:
            for line in data:
                fdata.write(line)

def loadSmallDatasets():
    #Load the smaller, well known datasets
    import pandas as pd
    import numpy as np

    data = datasets.load_iris()
    df = pd.DataFrame(data.data)
    df.drop_duplicates(inplace=True)
    np.save("datasets/iris.npy", df.to_numpy())

    # Load wine dataset
    data = datasets.load_wine()
    # datasets_dict['wine'] = (data.data, data.target)
    np.save("datasets/wine.npy", data.data)

    # Load swiss roll dataset
    X, Y = datasets.make_swiss_roll(n_samples=1500)
    np.save("datasets/swissroll.npy", X)

    # Load penguins dataset
    data = sns.load_dataset('penguins').dropna(thresh=6)
    cols_num = ['bill_length_mm', 'bill_depth_mm',
                'flipper_length_mm', 'body_mass_g']
    X = data[cols_num]
    Y = data['species'].astype('category').cat.codes
    np.save("datasets/penguins.npy", X)

    # Load auto-mpg dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    data = pd.read_csv(url, delim_whitespace=True)
    data.columns = ['mpg', 'cylinders', 'displacement', 'horsepower',
                    'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    data.horsepower = pd.to_numeric(data.horsepower, errors='coerce')
    data = data.drop(['model_year', 'origin', 'car_name'], axis=1)
    data = data[data.horsepower.notnull()]
    X = data[['acceleration', 'cylinders',
                'displacement', 'horsepower', 'weight']]
    Y = data['mpg']
    np.save("datasets/auto-mpg.npy", X)

    # Load s-curve dataset
    X, Y = datasets.make_s_curve(n_samples=1500)
    np.save("datasets/s-curve.npy", X)

if __name__ == "__main__":
    loadEspadatoDatasets()
    loadSmallDatasets()