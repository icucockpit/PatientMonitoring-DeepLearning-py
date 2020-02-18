import numpy as np
import h5py

import load_data


class ICU():

    def __init__(self, config_reader):
        self.config_reader = config_reader
        self.path_to_encoded_tiphog_features_dir = config_reader.get("path_to_tiphog_features_dir")

    def load_features(self, features_file_path, labels_file_path, folder_path):
        """Loads features and labels for each csv file in features_file_path and labels_file_path

                   # Arguments
                       features_file_path: path to the file containing csv files for features
                       labels_file_path: path to the file containing files for labels
                       folder_path: path to the folder containing features and labels

                   # Returns
                       numpy.ndarray: all data.

                   # Raises
           """
        all_x_data = load_data.load_numeric_data_from_csv_using_a_file(features_file_path, folder_path)
        all_y_data = load_data.load_numeric_data_from_csv_using_a_file(labels_file_path, folder_path)

        return all_x_data, all_y_data

    def load_features_random_split(self, features_file_path, labels_file_path, folder_path, percentage_split):
        """Loads features and labels for each csv file in features_file_path and labels_file_path and splits the data
        into training and testing based on the percentage_split

                   # Arguments
                       features_file_path: path to the file containing csv files for features
                       labels_file_path: path to the file containing files for labels
                       folder_path: path to the folder containing features and labels
                       percentage_split: percentage used to split the data between testing and training

                   # Returns
                       numpy.ndarray: all data.

                   # Raises
        """

        all_x_data, all_y_data = self.load_features(features_file_path, labels_file_path, folder_path)

        assert all_x_data.shape[0] == all_y_data.shape[0]
        idx = np.random.permutation(all_x_data.shape[0])
        all_x_data, all_y_data = all_x_data[idx, :], all_y_data[idx, :]
        split_slice = round(all_x_data.shape[0] * percentage_split / 100)

        return {
            "x_train": all_x_data[:split_slice, :],
            "y_train": all_y_data[:split_slice, :],
            "x_test": all_x_data[split_slice:, :],
            "y_test": all_y_data[split_slice:, :]}

    def load_tip_hog_features(self, random_split=None):
        """Loads tip hog features
                   # Returns
                       numpy.ndarray: all data.
        """
        features_file = self.config_reader.get("path_to_tip_hog_features_file")
        labels_file = self.config_reader.get("path_to_tip_hog_labels_file")
        directory_name = self.config_reader.get("path_to_tiphog_features_labels_dir")

        if random_split:
            percentage_split = self.config_reader.get("percentage_training")
            return self.load_features_random_split(features_file, labels_file, directory_name, percentage_split)
        else:
            return self.load_features(features_file, labels_file, directory_name)

    def extract_labels(self, y):
        f = h5py.File(self.config_reader.get("path_to_tip_hog_features_file"), 'r')
        dset_y_data = f['y_data_enc']

        return dset_y_data[:, 2]



