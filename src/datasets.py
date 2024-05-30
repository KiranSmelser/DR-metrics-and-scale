import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
from tensorflow.keras.datasets import mnist, fashion_mnist


def load_datasets():
    datasets_dict = {}

    # Load iris dataset
    data = datasets.load_iris()
    df = pd.DataFrame(data.data)
    df.drop_duplicates(inplace=True)
    datasets_dict['iris'] = (df.to_numpy(), data.target)

    # Load wine dataset
    data = datasets.load_wine()
    datasets_dict['wine'] = (data.data, data.target)

    # Load swiss roll dataset
    X, Y = datasets.make_swiss_roll(n_samples=1500)
    datasets_dict['roll'] = (X, Y)

    # Load penguins dataset
    data = sns.load_dataset('penguins').dropna(thresh=6)
    cols_num = ['bill_length_mm', 'bill_depth_mm',
                'flipper_length_mm', 'body_mass_g']
    X = data[cols_num]
    Y = data['species'].astype('category').cat.codes
    datasets_dict['penguins'] = (X, Y)

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
    datasets_dict['auto'] = (X.drop_duplicates(), Y)

    # Load s-curve dataset
    X, Y = datasets.make_s_curve(n_samples=1500)
    datasets_dict['curve'] = (X, Y)

    # Load the bank dataset
    X, Y = pd.read_csv(
        '../../data/bank/X.csv'), pd.read_csv('../../data/bank/Y.csv')
    datasets_dict['bank'] = (X.drop_duplicates(), Y)

    # Load the CNAE-9 dataset
    X, Y = pd.read_csv(
        '../../data/cnae-9/X.csv'), pd.read_csv('../../data/cnae-9/Y.csv')
    datasets_dict['cnae9'] = (X.drop_duplicates(), Y)

    # Load the COIL-20 dataset
    X, Y = pd.read_csv(
        '../../data/coil-20/X.csv'), pd.read_csv('../../data/coil-20/Y.csv')
    datasets_dict['coil20'] = (X.drop_duplicates(), Y)

    # Load the Epileptic Seizure Recognition dataset
    X, Y = pd.read_csv(
        '../../data/epileptic/X.csv'), pd.read_csv('../../data/epileptic/Y.csv')
    datasets_dict['epilepsy'] = (X.drop_duplicates(), Y)

    # Load the Spambase dataset
    X, Y = pd.read_csv(
        '../../data/spambase/X.csv'), pd.read_csv('../../data/spambase/Y.csv')
    datasets_dict['spambase'] = (X.drop_duplicates(), Y)

    # Load the Human Activity Recognition Using Smartphones dataset
    X, Y = pd.read_csv(
        '../../data/har/X.csv'), pd.read_csv('../../data/har/Y.csv')
    datasets_dict['har'] = (X.drop_duplicates(), Y)

    # Load the Sentiment Labelled Sentences dataset
    X, Y = pd.read_csv(
        '../../data/sentiment/X.csv'), pd.read_csv('../../data/sentiment/Y.csv')
    datasets_dict['sentiment'] = (X.drop_duplicates(), Y)

    return datasets_dict


def load_big_datasets():
    datasets_dict = {}

    # Load the MNIST dataset
    (X_mnist, y_mnist), _ = mnist.load_data()

    # Flatten the images and normalize to the range [0, 1]
    X_mnist = X_mnist.reshape((X_mnist.shape[0], -1)) / 255.

    # Randomly select 10000 samples from the dataset
    indices_mnist = np.random.choice(
        X_mnist.shape[0], size=10000, replace=False)

    X, Y = X_mnist[indices_mnist], y_mnist[indices_mnist]
    datasets_dict['mnist'] = (X, Y)

    # Load the Fashion MNIST dataset
    (X_fmnist, y_fmnist), _ = fashion_mnist.load_data()

    # Flatten the images and normalize to the range [0, 1]
    X_fmnist = X_fmnist.reshape((X_fmnist.shape[0], -1)) / 255.

    # Randomly select 10000 samples from the dataset
    indices_fmnist = np.random.choice(
        X_fmnist.shape[0], size=10000, replace=False)

    X, Y = X_fmnist[indices_fmnist], y_fmnist[indices_fmnist]
    datasets_dict['fmnist'] = (X, Y)

    # Load the Spambase dataset
    X_spam, y_spam = pd.read_csv(
        '../../data/spambase/X.csv'), pd.read_csv('../../data/spambase/Y.csv')

    # Randomly select 1000 samples from the dataset
    indices_spam = np.random.choice(
        X_spam.shape[0], size=1000, replace=False)

    X, Y = X_spam.iloc[indices_spam, :], y_spam.iloc[indices_spam, :]
    datasets_dict['spambase'] = (X.drop_duplicates(), Y)

    return datasets_dict


def find_range(dataset):
    ranges = pd.read_csv('../ranges.csv')
    max = ranges.loc[ranges['dataset'] == dataset, 'max'].iloc[0]
    return max + 0.2
