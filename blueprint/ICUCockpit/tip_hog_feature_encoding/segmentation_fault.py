#!/usr/bin/env python3

import numpy as np # numpy...
# from sklearn.externals import joblib # for persistence (loading, storing the GMM model)
# import argparse # command line parsing
# import pandas as pd # for loading data from e.g. csv files
# import h5py
import time

# Get histogram
# -----------------------------------------------------------------
def get_histogram(n_clusters):
    histogram = np.array(np.zeros(n_clusters))
    #return histogram.astype('float32')
    return histogram
# -----------------------------------------------------------------


N = 9000000
all_k_hist = []
row = 0

print("Encoding...")
t0 = time.time()
while row < N:
    if row % 100000 == 0:
        print("row:", row, end='\r', flush=True)

    k_hist = get_histogram(2048)
    # print("type(k_hist)", type(k_hist))
    # print("k_hist.shape", k_hist.shape)
    # print("k_hist.dtype", k_hist.dtype)
    all_k_hist.append(k_hist)

    row += 1

t1 = time.time()
print("Elapsed time={:.2f} s".format(t1 - t0))


