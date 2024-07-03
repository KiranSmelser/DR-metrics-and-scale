# stress-metrics-and-scale

Repository accompanying the paper, "Normalized Stress is Not Normalized: How to Interpret Stress Correctly."

## Contents

- `metrics.py`: Python module for computing various stress metrics between high-dimensional and low-dimensional data. Includes an implementation of scale-normalized stress.
- `compute_metrics.py`: Script for computing stress metrics on dimensionality reduction (DR) results.
- `populate_datasets.py`: Script to load and save all datasets used in the experiment.
- `compute_embeddings.py`: Script for computing and saving embeddings using DR techniques for all datasets.
- `analysis.ipynb`: Jupyter notebook for analyzing results and generating tables.
- `viz.ipynb`: Jupyter notebook for creating the figures seen in the paper.
- `figs/`: Folder containing the figures seen in the paper.

## Requirements

The code in this repository was written using Python version 3.12 and utilizes the following libraries:
- json
- pandas
- numpy
- scipy
- tqdm
- warnings
- sklearn
- umap
- os
- pylab
- zadu
- seaborn
- itertools
- matplotlib
- urllib.request

## Usage

1. Clone this repository: `git clone https://github.com/KiranSmelser/stress-metrics-and-scale`
2. Install the required libraries: `pip install -r requirements.txt`
3. Run the scripts and notebooks as needed.

<!-- ## Citation

If you find this work useful, please consider citing our paper: