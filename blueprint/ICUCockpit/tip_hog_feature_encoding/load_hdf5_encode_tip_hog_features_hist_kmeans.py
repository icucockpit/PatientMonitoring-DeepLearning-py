#!/usr/bin/env python3
import numpy as np # numpy...
from sklearn.externals import joblib # for persistence (loading, storing the GMM model)
import argparse # command line parsing
import pandas as pd # for loading data from e.g. csv files
import h5py
import time
import faulthandler

def get_histogram(n_clusters, kmeans_predictions):
    no_of_predictions = len(kmeans_predictions)
    histogram = np.zeros(n_clusters)
    for prediction in kmeans_predictions:
        histogram[prediction] += 1
    histogram /= no_of_predictions # normalize to sum = 1
    return histogram

def append_to_hdf5_file(hdf5_file_object, x_data_enc, y_data_enc, frames_data_enc):
    #check if datasets exist
    if 'x_data_enc' in list(hdf5_file_object.keys()):
        # Resize and add data
        #print("Resize and add data...")
        hdf5_file_object["x_data_enc"].resize((hdf5_file_object["x_data_enc"].shape[0] + x_data_enc.shape[0]), axis=0)
        hdf5_file_object["x_data_enc"][-x_data_enc.shape[0]:] = x_data_enc
    else:
        # Create resizable dataset
        print("Creating dataset: x_data_enc")
        hdf5_file_object.create_dataset('x_data_enc', dtype=x_data_enc.dtype, data=x_data_enc, chunks=True, maxshape=(None, x_data_enc.shape[1]))
    #check if datasets exist
    if 'y_data_enc' in list(hdf5_file_object.keys()):
        # Resize and add data
        #print("Resize and add data...")
        hdf5_file_object["y_data_enc"].resize((hdf5_file_object["y_data_enc"].shape[0] + y_data_enc.shape[0]), axis=0)
        hdf5_file_object["y_data_enc"][-y_data_enc.shape[0]:] = y_data_enc
    else:
        # Create resizable dataset
        print("Creating dataset: y_data_enc")
        hdf5_file_object.create_dataset('y_data_enc', dtype=y_data_enc.dtype, data=y_data_enc, chunks=True, maxshape=(None, y_data_enc.shape[1]))
    #check if datasets exist
    if 'frames_data_enc' in list(hdf5_file_object.keys()):
        # Resize and add data
        #print("Resize and add data...")
        hdf5_file_object["frames_data_enc"].resize((hdf5_file_object["frames_data_enc"].shape[0] + frames_data_enc.shape[0]), axis=0)
        hdf5_file_object["frames_data_enc"][-frames_data_enc.shape[0]:] = frames_data_enc
    else:
        # Create resizable dataset
        print("Creating dataset: frames_data_enc")
        hdf5_file_object.create_dataset('frames_data_enc', dtype=frames_data_enc.dtype, data=frames_data_enc, chunks=True, maxshape=(None,))

def main():
    faulthandler.enable()

    parser = argparse.ArgumentParser(description='Encode features based on kmeans cluster model.')
    parser.add_argument('-inhdf5file', '--in_hdf5_file_name', required=True, help='...')
    parser.add_argument('-kmeansm', '--inputkmeans', required=True, help='Input Kmeans model.')
    parser.add_argument('-wf', '--windowf', type=int, required=True, help='Window in frames.')
    parser.add_argument('-sf', '--stepf', type=int, required=True, help='Step in frames.')
    args = parser.parse_args()
    in_hdf5_file_name = args.in_hdf5_file_name
    print("in_hdf5_file_name:", in_hdf5_file_name)
    print("Kmeans file:", args.inputkmeans)

    # Open input hfd5 file
    print("Opening", in_hdf5_file_name)
    f_in = h5py.File(in_hdf5_file_name, 'r')
    print("keys:", list(f_in.keys()))
    x_data = f_in['x_data']
    y_data = f_in['y_data']
    frames_data = np.array(f_in['frames_data']) #np array to make things faster

    print("type(x_data)", type(x_data))
    print("x_data.shape", x_data.shape)
    print("x_data.dtype", x_data.dtype)

    print("type(y_data)", type(y_data))
    print("y_data.shape", y_data.shape)
    print("y_data.dtype", y_data.dtype)

    print("type(frames_data)", type(frames_data))
    print("frames_data.shape", frames_data.shape)
    print("frames_data.dtype", frames_data.dtype)

    # Load Kmeans model
    print("Loading Kmeans model...")
    kmeans_input_file = args.inputkmeans
    kmeans_estimator = joblib.load(kmeans_input_file)
    kmeans_params=kmeans_estimator.get_params()
    n_clusters = kmeans_params['n_clusters']
    print("n_clusters", n_clusters)

    N = x_data.shape[0]
    print("rows N =", N)

    window_in_frames = args.windowf
    print("window_in_frames =", window_in_frames)
    assert window_in_frames != 0 and window_in_frames < N
    assert (window_in_frames % 2) != 0

    step_in_frames = args.stepf
    print("step_in_frames =", step_in_frames)
    assert step_in_frames != 0 and step_in_frames < N

    # Create out file object
    output_file = in_hdf5_file_name + "_w" + str(window_in_frames) + "_s" + str(step_in_frames) + "_ENCODED.hdf5"
    f_out = h5py.File(output_file, "w")

    row_start = 0
    row_end = 0
    frame_end = 0
    row = 0
    print_counter = 0
    all_k_hist = []
    all_frame_in_the_middle = []
    all_classg = []

    encoding_batch_size = 3000
    encoding_batch_counter = 0

    print("Encoding...")
    t0 = time.time()
    #N = 400000
    while row < N:
        # Group according to frame number (there are multiple rows in x_data for one frame)
        frame_start = frames_data[row_start]

        row = row_start
        while (row < N) and (frames_data[row] < (frame_start + window_in_frames)):
            if (print_counter * step_in_frames) == 10000:
                print_counter = 0
                print("row", row, "frame", frames_data[row], end='\r', flush=True)
            row += 1
        row_end = row - 1

        frame_end = frames_data[row_end]
        frame_at_window_center = frame_start + round((frame_end - frame_start) / 2)

        # Processing code:
        # ----------------------------------------------------------
        # ----------------------------------------------------------

        if row_start == row_end:
            x_data_instance = x_data[row_start, :]
            classg = y_data[row_start, :]
        else:
            x_data_instance = x_data[row_start:row_end, :]
            # Find row of frame_at_window_center of window to get corresponding labels
            for r in range(row_start, row_end):
                if frames_data[r] == frame_at_window_center:
                    classg = y_data[r, :]
                    break

        # Calculate encoding

        #a)
        kmeans_pred = kmeans_estimator.predict(np.atleast_2d(x_data_instance))
        k_hist = get_histogram(n_clusters, kmeans_pred)

        #b)
        k_hist_copy = np.copy(k_hist)
        all_k_hist.append(k_hist_copy)
        all_frame_in_the_middle.append(np.copy(frame_at_window_center))
        all_classg.append(np.copy(classg))

        # ----------------------------------------------------------
        # ----------------------------------------------------------


        next_frame_start = frame_start + step_in_frames
        if next_frame_start > frames_data[-1]:
            break
        # Find row of first instance of next frame
        # range([start], stop[, step])
        # The last integer generated by range() is up to, but not including, stop
        for r in range(row_start, N):
            if frames_data[r] == next_frame_start:
                row_start = r
                break
        print_counter += 1

        encoding_batch_counter += 1
        if encoding_batch_counter == encoding_batch_size:
            encoding_batch_counter = 0
            k_hist2dc = np.array(all_k_hist)
            k_hist2dc = k_hist2dc.astype('float32')
            all_frame_in_the_middle2dc = np.array(all_frame_in_the_middle)
            all_frame_in_the_middle2dc = all_frame_in_the_middle2dc.astype('int32')
            classg2dc = np.array(all_classg)
            classg2dc = classg2dc.astype('S')
            # print("k_hist2dc: ", k_hist2dc.shape)
            # print("all_frame_in_the_middle2dc: ", all_frame_in_the_middle2dc.shape)
            # print("classg2dc: ", classg2dc.shape)
            append_to_hdf5_file(f_out, k_hist2dc, classg2dc, all_frame_in_the_middle2dc)
            all_k_hist = []
            all_frame_in_the_middle = []
            all_classg = []


    print("last row index", row-1, "last frame", frames_data[row-1], end='\r', flush=True)
    # Append last batch:
    if encoding_batch_counter > 0:
        encoding_batch_counter = 0
        k_hist2dc = np.array(all_k_hist)
        k_hist2dc = k_hist2dc.astype('float32')
        all_frame_in_the_middle2dc = np.array(all_frame_in_the_middle)
        all_frame_in_the_middle2dc = all_frame_in_the_middle2dc.astype('int32')
        classg2dc = np.array(all_classg)
        classg2dc = classg2dc.astype('S')
        # print("k_hist2dc: ", k_hist2dc.shape)
        # print("all_frame_in_the_middle2dc: ", all_frame_in_the_middle2dc.shape)
        # print("classg2dc: ", classg2dc.shape)
        append_to_hdf5_file(f_out, k_hist2dc, classg2dc, all_frame_in_the_middle2dc)
        all_k_hist = []
        all_frame_in_the_middle = []
        all_classg = []

    t1 = time.time()
    print("\nElapsed time={:.2f} s".format(t1 - t0))

    # Close out file
    f_out.close() # flush() is done automatically
    print("Annotation saved to:", output_file)
    print("Done")

    print("Contents of output file:")
    f = h5py.File(output_file, 'r')
    print("keys:", list(f.keys()))
    dset_x_data = f['x_data_enc']
    dset_y_data = f['y_data_enc']
    dset_frames_data = f['frames_data_enc']

    print("dset_x_data.shape", dset_x_data.shape)
    print("dset_x_data.dtype", dset_x_data.dtype)

    print("dset_y_data.shape", dset_y_data.shape)
    print("dset_y_data.dtype", dset_y_data.dtype)

    print("dset_frames_data.shape", dset_frames_data.shape)
    print("dset_frames_data.dtype", dset_frames_data.dtype)

if __name__ == '__main__':
    main()
