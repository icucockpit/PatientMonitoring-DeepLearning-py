#!/usr/bin/env python3

import numpy as np # numpy...
import pdb # for debugging
import argparse # command line parsing
from sklearn.cluster import KMeans
from sklearn.externals import joblib # for persistence (loading, storing the GMM model)
import h5py
import random
random.seed(0)

# parse command line
parser = argparse.ArgumentParser(description='...')
parser.add_argument('-inhdf5file', '--in_hdf5_file_name', required=True, help='...')
parser.add_argument('-cmodel', '--cluster_model', required=True, help='...')
parser.add_argument('-k', '--k_clusters', type=int ,required=True, help='...')
args = parser.parse_args()
in_hdf5_file_name = args.in_hdf5_file_name
print("in_hdf5_file_name:", in_hdf5_file_name)
print("cluster_model:", args.cluster_model)
print("n_clusters:", args.k_clusters)

# Open input hfd5 file
print("Opening", in_hdf5_file_name)
f_in = h5py.File(in_hdf5_file_name, 'r')
print("keys:", list(f_in.keys()))
x_data = f_in['x_data']

print("x_data.shape", x_data.shape)
print("x_data.dtype", x_data.dtype)

n_clusters = args.k_clusters
print("n_clusters =", n_clusters)
print("K-means clustering on x_data...")
kmeans_estimator = KMeans(n_clusters, n_jobs = 20, random_state=0, verbose=1, max_iter=1200)
kmeans_estimator.fit(x_data)

# persistence
output_file = args.cluster_model
joblib.dump(kmeans_estimator, output_file)
print("kmeans_estimator saved to:", output_file)
