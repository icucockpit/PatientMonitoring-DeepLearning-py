#!/usr/bin/env python3

import numpy as np # numpy...
import pdb # for debugging
import argparse # command line parsing
import pandas as pd # for loading data from e.g. csv files

import h5py

# parse command line
parser = argparse.ArgumentParser(description='...')
parser.add_argument('-i', '--input_csv_file', required=True, help='...')
parser.add_argument('-o', '--out_hdf5_file', required=True, help='...')

args = parser.parse_args()
print("input_csv_file:", args.input_csv_file, flush=True)
print("out_hdf5_file:", args.out_hdf5_file, flush=True)

print("Loading csv file...", flush=True)
df = pd.read_csv(args.input_csv_file, delimiter = ",", header = None)
df_num = df.select_dtypes(include=[np.number])
all_num_data = df_num.values
df_str = df.select_dtypes(include=['object'])
all_str_data = df_str.values

print("type(all_num_data):", type(all_num_data), flush=True)
print("all_num_data.dtype:", all_num_data.dtype, flush=True)
print("all_num_data.shape:", all_num_data.shape, flush=True)

print("type(all_str_data):", type(all_str_data), flush=True)
print("all_str_data.dtype:", all_str_data.dtype, flush=True)
print("all_str_data.shape:", all_str_data.shape, flush=True)

# Select columns with features
x_data = all_num_data[:, 3:]  # select columns 3 through end
x_data = x_data.astype('float32')
print("type(x_data):", type(x_data), flush=True)
print("x_data.dtype:", x_data.dtype)
print("x_data.shape:", x_data.shape)

# Select columns with frame numbers
meta_data = all_num_data[:, :3]   # select column 0-2, additional info (frame etc.)
frames_data = meta_data[:, 2]
frames_data = frames_data.astype('int32')
print("type(frames_data):", type(frames_data), flush=True)
print("frames_data.dtype: ", frames_data.dtype, flush=True)
print("frames_data.shape: ", frames_data.shape, flush=True)

# Convert columns with labels
#y_data = np.concatenate(all_str_data, axis=0)
y_data = all_str_data.astype('S')
print("type(y_data):", type(y_data), flush=True)
print("y_data.dtype: ", y_data.dtype, flush=True)
print("y_data.shape: ", y_data.shape, flush=True)


print("Saving uncompressed hdf5 file...", flush=True)
f = h5py.File(args.out_hdf5_file, "w")
f.create_dataset("x_data", dtype=x_data.dtype, data=x_data)
f.create_dataset("y_data", dtype=y_data.dtype, data=y_data)
f.create_dataset("frames_data", dtype=frames_data.dtype, data=frames_data)
f.close() # flush() is done automatically


print("Loading uncompressed hdf5 file for a check...", flush=True)
f = h5py.File(args.out_hdf5_file, 'r')
print("keys:", list(f.keys()))
dset_x_data = f['x_data']
dset_y_data = f['y_data']
dset_frames_data = f['frames_data']

print("dset_x_data.shape", dset_x_data.shape)
print("dset_x_data.dtype", dset_x_data.dtype)

print("dset_y_data.shape", dset_y_data.shape)
print("dset_y_data.dtype", dset_y_data.dtype)

print("dset_frames_data.shape", dset_frames_data.shape)
print("dset_frames_data.dtype", dset_frames_data.dtype)

