#!/usr/bin/env python3

import numpy as np
import argparse # command line parsing
from sklearn.cluster import AgglomerativeClustering
from sklearn.externals import joblib # for persistence (loading, storing the GMM model)
import h5py
import ntpath
import time

from itertools import groupby
from operator import itemgetter


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# Parse command line
parser = argparse.ArgumentParser(description='...')
parser.add_argument('-inhdf5file', '--in_hdf5_file_name', required=True, help='...')
args = parser.parse_args()
in_hdf5_file_name = args.in_hdf5_file_name
print("**************************************************************************", flush=True)
print("in_hdf5_file_name:", in_hdf5_file_name, flush=True)


# Open input hfd5 file
print("Opening", in_hdf5_file_name, flush=True)
f_in = h5py.File(in_hdf5_file_name, 'r')
print("keys:", list(f_in.keys()), flush=True)

x_data = f_in['x_data_enc']
y_data = f_in['y_data_enc']
frames_data = f_in['frames_data_enc']

print("type(x_data)", type(x_data), flush=True)
print("x_data.dtype", x_data.dtype, flush=True)
print("x_data.shape", x_data.shape, flush=True)
print("type(y_data)", type(y_data), flush=True)
print("y_data.dtype", y_data.dtype, flush=True)
print("y_data.shape", y_data.shape, flush=True)
print("type(frames_data)", type(frames_data), flush=True)
print("frames_data.dtype", frames_data.dtype, flush=True)
print("frames_data.shape", frames_data.shape, flush=True)


t0 = time.time()


# Analyze features
# find unique
from sklearn.preprocessing import LabelBinarizer
# print("y_data[0,:]\n", y_data[0,:], flush=True)
# print("y_data.shape", y_data.shape, flush=True)

# Take only label columns (First column has timestamp, second column indicates whether a point
# is within a predefined ROI or not)
nda_y_data = np.array(y_data[:,2:])
print("type(nda_y_data)", type(nda_y_data), flush=True)
print("nda_y_data.shape", nda_y_data.shape, flush=True)
# Find unique feature labels in each clomun.
# The same label may not always be on one column only, this is due to the way tip-hog features are being extracted.
encoder = LabelBinarizer()
columns_to_delete = []
for column_index, column in enumerate(nda_y_data.T):
    encoder.fit(column)
    print("unique labels in column", column_index, ":", encoder.classes_)
    if encoder.classes_[0] ==  b'no_label':
        columns_to_delete.append(column_index)


# Delete columns that have only 'no_label'
nda_y_data = np.delete(nda_y_data, columns_to_delete, axis=1)
print("nda_y_data.shape", nda_y_data.shape, flush=True)


if nda_y_data.shape[1] > 1:
    #print("unique labels in row", np.where(nda_y_data[:,1] == b'no_label')[0])
    nda_y_data[np.where(nda_y_data[:,1:] == b'no_label')[0],1] = b'x'
    #print("nda_y_data", nda_y_data[0:10,:] )

nda_all_labels = nda_y_data.astype('str')

label_list = ['EEG: 00', 'EEG: 01', 'EEG: 02', 'EEG: 03', 'EEG: 04', 'EEG: 05', 'no_label']

list_y_data_timestamp = []
# Save as hdf5
output_file = in_hdf5_file_name + "_continuous_data.hdf5"
out_file_object = h5py.File(output_file, "w")
for label in label_list:
    # Get sorted unique indices of rows where the label appears (any column)
    label_row_index = np.unique(np.where(nda_all_labels == label)[0])
    if label_row_index.size != 0:
        grp = out_file_object.create_group(label)
        counter = 0
        # Creates a list of lists of continuous labels for label_row_index
        for k, g in groupby(enumerate(label_row_index), lambda ix : ix[0] - ix[1]):
            print("label:", label)
            list_continuous_labels = list(map(itemgetter(1), g))
            print("list_continuous_labels[0] = ", list_continuous_labels[0])
            print("list_continuous_labels[-1] = ", list_continuous_labels[-1])
            print("x_data[list_continuous_labels,:].shape", x_data[list_continuous_labels,:].shape, flush=True)
            x_dataset_name = path_leaf(in_hdf5_file_name) + "_x" + str(counter)
            time_dataset_name = path_leaf(in_hdf5_file_name) + "_time" + str(counter)
            print("x_dataset_name, time_dataset_name:", x_dataset_name, time_dataset_name)
            grp.create_dataset(x_dataset_name, data=x_data[list_continuous_labels, :])
            print("dtype", y_data[list_continuous_labels, 0][0].dtype)
            grp.create_dataset(time_dataset_name, data=y_data[list_continuous_labels, 0], dtype="S26")
            counter += 1

        print("grp.keys():", grp.keys())
        print("grp.values():", grp.values())

out_file_object.close() # flush() is done automatically
print("Annotation saved to:", output_file)
print("Done")

t1 = time.time()
print("Elapsed time={:.2f} s".format(t1 - t0))

# Check saved hfd5 file
print("Opening", output_file, flush=True)
f_in = h5py.File(output_file, 'r')

# Recursively visit all objects in this group and subgroups.
def printname_keys(name):
    print(name)

f_in.visit(printname_keys)
