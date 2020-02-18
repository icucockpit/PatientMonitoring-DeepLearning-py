#!/usr/bin/env python3

import numpy as np # numpy...
import pdb # for debugging
import argparse # command line parsing
import pandas as pd # for loading data from e.g. csv files
import h5py
import random
random.seed(0)

# parse command line
parser = argparse.ArgumentParser(description='...')
parser.add_argument('-iflist', '--input_file_list', required=True, help='...')
parser.add_argument('-ifpath', '--input_file_path', required=True, help='...')
parser.add_argument('-o', '--output_hdf5_file', required=True, help='...')
parser.add_argument('-n', '--n_samples', type=int, required=True, help='...')

args = parser.parse_args()
print("input_file_list:", args.input_file_list)
print("input_file_path:", args.input_file_path)
print("output_hdf5_file:", args.output_hdf5_file)

n_samples = args.n_samples
print("n_samples:", n_samples)

# comma delimited is the default
print("Loading file list...")
df = pd.read_csv(args.input_file_list, header = None)
df_str = df.select_dtypes(include=['object'])
list_of_filenames = df_str.values

print("Loading all data...")
all_x_data = []
count_x = 0
count_y = 0
for file_name in list_of_filenames:
    full_path=str(file_name[0])
    print("Loading file:", full_path)

    # Open input hfd5 file
    # Open input hfd5 file
    print("Opening", full_path)
    f_in = h5py.File(full_path, 'r')
    print("keys:", list(f_in.keys()))
    x_data = f_in['x_data']
    y_data = f_in['y_data']
    frames_data = f_in['frames_data']
    
    count_x = count_x + len(x_data)
    count_y = count_y + len(y_data)

    
    print("count_x != count_y", full_path, count_x, count_y)
    print("x_data.shape", x_data.shape)
    print("x_data.dtype", x_data.dtype)

    print("y_data.shape", y_data.shape)
    print("y_data.dtype", y_data.dtype)

    print("frames_data.shape", frames_data.shape)
    print("frames_data.dtype", frames_data.dtype)

    all_x_data.append(x_data)

all_x_data=np.concatenate(all_x_data, axis=0)
print("type(all_x_data.dtype)", type(all_x_data.dtype))
print("all_x_data.dtype", all_x_data.dtype)
print("all_x_data.shape", all_x_data.shape)

with h5py.File(args.output_hdf5_file, 'w') as out:
    N = all_x_data.shape[0]
    indexes = random.sample(range(0, N), n_samples)
    feed = np.take(all_x_data, indexes, axis=0)
    print("feed.shape:", feed.shape)
    out.create_dataset("x_data", data=feed)


f_in.close()

# Check:
# Open out_hdf5_file_name hfd5 file
print("Opening", args.output_hdf5_file)
f = h5py.File(args.output_hdf5_file, 'r')
print("keys:", list(f.keys()))
x_data = f['x_data']

print("x_data.shape", x_data.shape)
print("x_data.dtype", x_data.dtype)

f.close()
