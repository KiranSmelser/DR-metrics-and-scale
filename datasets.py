import pandas as pd
from sklearn import datasets
import seaborn as sns



def load_datasets():
    datasets_dict = {}

    # Load iris dataset
    data = datasets.load_iris()
    datasets_dict['iris'] = (data.data, data.target)

    # Load wine dataset
    data = datasets.load_wine()
    datasets_dict['wine'] = (data.data, data.target)

    # Load swiss roll dataset
    X, Y = datasets.make_swiss_roll(n_samples=200)
    datasets_dict['swiss_roll'] = (X, Y)

    # Load penguins dataset
    data = sns.load_dataset('penguins').dropna(thresh = 6)
    cols_num = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X = data[cols_num]
    Y = data['species'].astype('category').cat.codes
    datasets_dict['penguins'] = (X, Y)

    # Load auto-mpg dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    data = pd.read_csv(url, delim_whitespace=True)
    data.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    data.horsepower = pd.to_numeric(data.horsepower, errors='coerce')
    data = data.drop(['model_year', 'origin', 'car_name'], axis=1)
    data = data[data.horsepower.notnull()]
    X = data[['acceleration', 'cylinders', 'displacement', 'horsepower', 'weight']]
    Y = data['mpg']
    datasets_dict['auto_mpg'] = (X, Y)

    # Load s-curve dataset
    X, Y = datasets.make_s_curve(n_samples=200)
    datasets_dict['s_curve'] = (X, Y)

    # Load the bank dataset
    X, Y = pd.read_csv('../data/bank/X.csv'), pd.read_csv('../data/bank/Y.csv')
    datasets_dict['bank'] = (X, Y)

    # Load the CNAE-9 dataset
    X, Y = pd.read_csv(
        '../data/cnae-9/X.csv'), pd.read_csv('../data/cnae-9/Y.csv')
    datasets_dict['cnae9'] = (X, Y)

    # Load the COIL-20 dataset
    X, Y = pd.read_csv(
        '../data/coil-20/X.csv'), pd.read_csv('../data/coil-20/Y.csv')
    datasets_dict['coil20'] = (X, Y)

    # Load the Epileptic Seizure Recognition dataset
    X, Y = pd.read_csv(
        '../data/epileptic/X.csv'), pd.read_csv('../data/epileptic/Y.csv')
    datasets_dict['epilepsy'] = (X, Y)

    # Load the Spambase dataset
    X, Y = pd.read_csv(
        '../data/spambase/X.csv'), pd.read_csv('../data/spambase/Y.csv')
    datasets_dict['spambase'] = (X, Y)

    # Load the Human Activity Recognition Using Smartphones dataset
    X, Y = pd.read_csv(
        '../data/har/X.csv'), pd.read_csv('../data/har/Y.csv')
    datasets_dict['har'] = (X, Y)

    # Load the Sentiment Labelled Sentences dataset
    X, Y = pd.read_csv(
        '../data/sentiment/X.csv'), pd.read_csv('../data/sentiment/Y.csv')
    datasets_dict['sentiment'] = (X, Y)

    return datasets_dict
