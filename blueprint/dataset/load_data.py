import numpy as np  # numpy
import pandas as pd  # for loading data from e.g. csv files
import math
from sklearn.preprocessing import LabelBinarizer


def load_numeric_data_from_csv_file(path_to_file):
    """Loads numeric data from a csv file.

        # Arguments
            file_name: full path to csv file.

        # Returns
            numpy.ndarray: all numeric data in the file.

        # Raises

        """
    df = pd.read_csv(path_to_file, delimiter=",", header=None)
    df_num = df.select_dtypes(include=[np.number])

    return df_num.values


def load_object_data_from_csv_file(path_to_file):
    """Loads object data from a csv file. This is used for loading labels.

        # Arguments
            file_name: full path to csv file.

        # Returns
            numpy.ndarray: all object data in the file.

        # Raises

        Note that this will return all object dtype columns including strings
        """
    df = pd.read_csv(path_to_file, delimiter=",", header=None)
    df_obj = df.select_dtypes(include=['object'])

    return df_obj.values


def load_numeric_data_from_csv_file_list(path_to_data_dir, path_to_file_list):
    """Loads numeric data from each csv file in the list.

        # Arguments
            path_to_data_dir: path to csv files.
            path_to_file_list:

        # Returns
            numpy.ndarray: all data.

        # Raises

        """
    df = pd.read_csv(path_to_file_list, header=None)
    df_obj = df.select_dtypes(include=['object'])
    file_names = df_obj.values

    all_num_data = []
    for file_name in file_names:
        full_path = path_to_data_dir + '/' + str(file_name[0])
        data = load_numeric_data_from_csv_file(full_path)
        all_num_data.append(data)

    return np.concatenate(all_num_data, axis=0)


def load_object_data_from_csv_file_list(path_to_data_dir, path_to_file_list):
    """Loads object data from each csv file in the list.

        # Arguments
            path_to_data_dir: path to csv files.
            path_to_file_list:

        # Returns
            numpy.ndarray: all data.

        # Raises

        """
    df = pd.read_csv(path_to_file_list, header=None)
    df_obj = df.select_dtypes(include=['object'])
    file_names = df_obj.values

    all_obj_data = []
    for file_name in file_names:
        full_path = path_to_data_dir + '/' + str(file_name[0])
        data = load_object_data_from_csv_file(full_path)
        all_obj_data.append(data)

    return np.concatenate(all_obj_data, axis=0)


def load_numeric_data_from_csv_using_a_file(path_to_file, path_to_features_folder):
    """Loads object data from each csv file in the file.

            # Arguments
                path_to_file: path to file containing csv files.
                path_to_features_folder: absolute path to

            # Returns
                numpy.ndarray: all data.

            # Raises
    """
    features_file_list = load_object_data_from_csv_file(path_to_file)
    all_data = []
    for file in features_file_list:
        full_path = path_to_features_folder + '/' + file[0]
        x_data = load_numeric_data_from_csv_file(full_path)
        all_data.append(x_data)

    return np.concatenate(all_data, axis = 0)


def load_numeric_data_from_csv_file_percentage_split(path_to_file, percent_train):
    """Loads numeric data from a csv file and splits it into training and testing set.

        # Arguments
            path_to_file: Full path to csv file.
            percent_train: Percentage [%] to take for training.

        # Returns
            numpy.ndarray, numpy.ndarray: training set, testing set

        # Raises

        """
    data = load_numeric_data_from_csv_file(path_to_file)
    split_slice = round(data.shape[0] * percent_train / 100)

    return data[:split_slice, :], data[split_slice:, :]


def load_object_data_from_csv_file_percentage_split(path_to_file, percent_train):
    """Loads object data from a csv file and splits it into training and testing set.

        # Arguments
            path_to_file: Full path to csv file.
            percent_train: Percentage [%] to take for training.

        # Returns
            numpy.ndarray, numpy.ndarray: training set, testing set

        # Raises

        """
    data = load_object_data_from_csv_file(path_to_file)
    split_slice = round(data.shape[0] * percent_train / 100)

    return data[:split_slice, :], data[split_slice:, :]


def load_numeric_data_from_csv_file_list_percentage_split(path_to_data_dir, path_to_file_list, percent_train):
    """Loads numeric data from each csv file in the list and splits data from each file
        into a training and testing set. Training and testing set are concatenated separately

        # Arguments
            path_to_data_dir: path to csv files.
            path_to_file_list: txt file with list of csv files to open
            percent_train: percentage to keep for training

        # Returns
            Numpy arrays: training data, testing data.

        # Raises

        """
    df = pd.read_csv(path_to_file_list, header=None)
    df_str = df.select_dtypes(include=['object'])
    file_names = df_str.values

    all_num_data_train = []
    all_num_data_test = []
    for file_name in file_names:
        full_path = path_to_data_dir + '/' + str(file_name[0])
        num_data_train, num_data_test = load_numeric_data_from_csv_file_percentage_split(full_path, percent_train)
        all_num_data_train.append(num_data_train)
        all_num_data_test.append(num_data_test)

    return np.concatenate(all_num_data_train, axis=0), np.concatenate(all_num_data_test, axis=0)


def load_object_data_from_csv_file_list_percentage_split(path_to_data_dir, path_to_file_list, percent_train):
    """Loads object data from each csv file in the list and splits data from each file
        into a training and testing set. Training and testing set are concatenated separately

        # Arguments
            path_to_data_dir: path to csv files.
            path_to_file_list: txt file with list of csv files to open
            percent_train: percentage to take for training

        # Returns
            Numpy arrays: training data, testing data.

        # Raises

        """
    df = pd.read_csv(path_to_file_list, header=None)
    df_str = df.select_dtypes(include=['object'])
    file_names = df_str.values

    all_obj_data_train = []
    all_obj_data_test = []
    for file_name in file_names:
        full_path = path_to_data_dir + '/' + str(file_name[0])
        obj_data_train, obj_data_test = load_object_data_from_csv_file_percentage_split(full_path, percent_train)
        all_obj_data_train.append(obj_data_train)
        all_obj_data_test.append(obj_data_test)

    return np.concatenate(all_obj_data_train, axis=0), np.concatenate(all_obj_data_test, axis=0)


def load_dataset_from_csv_file_lists_percentage_split(path_to_features_data_dir, path_to_features_file_list, path_to_labels_data_dir, path_to_labels_file_list, percent_train):
    """Loads whole dataset (features and labels). Uses each csv file in the list and splits data from each file
        into a training and testing set. Training and testing set are concatenated separately
        
        # Arguments
            path_to_features_data_dir: path to csv files with features.
            path_to_features_file_list: txt file with list of csv files to open
            path_to_labels_data_dir: path to csv files with labels
            path_to_labels_file_list: txt file with list of csv files to open
            percent_train: percentage to take for training

        # Returns
            dictionary: {"x_train": , "y_train": ,
            "x_test": , "y_test": }

        # Raises

        """
    x_data_train, x_data_test = load_numeric_data_from_csv_file_list_percentage_split(path_to_features_data_dir, path_to_features_file_list, percent_train)
    y_data_train, y_data_test = load_object_data_from_csv_file_list_percentage_split(path_to_labels_data_dir, path_to_labels_file_list, percent_train)

    return {"x_train": x_data_train, "y_train": y_data_train,
            "x_test": x_data_test, "y_test": y_data_test}


def load_dataset_from_csv_file_lists_random_split(path_to_features_data_dir, path_to_features_file_list, path_to_labels_data_dir, path_to_labels_file_list, percent_train):
    """Loads whole dataset (features and labels). Splits all data from all files randomly
        into a training and testing set. Training and testing set are concatenated separately
        
        # Arguments
            path_to_features_data_dir: path to csv files with features.
            path_to_features_file_list: txt file with list of csv files to open
            path_to_labels_data_dir: path to csv files with labels
            path_to_labels_file_list: txt file with list of csv files to open
            percent_train: percentage to take for training

        # Returns
            dictionary: {"x_train": , "y_train": , "x_test": , "y_test": }

        # Raises

        """
    x_data = load_numeric_data_from_csv_file_list(path_to_features_data_dir, path_to_features_file_list)
    y_data = load_object_data_from_csv_file_list(path_to_labels_data_dir, path_to_labels_file_list)
    assert x_data.shape[0] == y_data.shape[0]
    idx = np.random.permutation(x_data.shape[0])
    x_data, y_data = x_data[idx,:], y_data[idx,:]

    split_slice = round(x_data.shape[0] * percent_train / 100)

    return {"x_train": x_data[:split_slice, :], "y_train": y_data[:split_slice, :],
            "x_test": x_data[split_slice:, :], "y_test": y_data[split_slice:, :]}


def preprocess_dataset_float32_binarize(data):
    """Dateset preprocessing: features into float33 and labels into binary 
        (binarize labels in a one-vs-all fashion)
        
        # Arguments
            data: {"x_train": , "y_train": ,
            "x_test": , "y_test": }

        # Returns
            data: {"x_train": , "y_train": ,
            "x_test": , "y_test": }

        # Raises

        """    
    x_data_train = data["x_train"].astype('float32')
    x_data_test = data["x_test"].astype('float32')

    all_y_data = np.concatenate((data["y_train"], data["y_test"]), axis=0)
    encoder = LabelBinarizer()
    encoder.fit(all_y_data)
    y_data_train_cat = encoder.transform(data["y_train"])
    y_data_test_cat = encoder.transform(data["y_test"])

    return {"x_train": x_data_train, "y_train": y_data_train_cat,
            "x_test": x_data_test, "y_test": y_data_test_cat}


def preprocess_dataset_float32_binarize_2(data):
    """Dateset preprocessing: features into float32 and labels into binary
        (binarize labels in a one-vs-all fashion)

        # Arguments
            data: {"x_train": , "y_train": }

        # Returns

        # Raises

        """
    data["x_train"] = data["x_train"].astype('float32')
    data["x_test"] = data["x_test"].astype('float32')

    all_y_data = np.concatenate((data["y_train"], data["y_test"]), axis=0)
    encoder = LabelBinarizer()
    encoder.fit(all_y_data)
    data["y_train"] = encoder.transform(data["y_train"])
    data["y_test"] = encoder.transform(data["y_test"])
    return encoder.classes_


def preprocess_dataset_in_list_float32_binarize(data):
    encoder = LabelBinarizer()
    train_labels = []
    x_data_train = []
    y_data_train = []
    for x_data, y_data in zip(data["x_train"], data["y_train"]):
        x_data_train.append(x_data.astype('float32'))
        encoder.fit(y_data)
        y_data_train.append(encoder.transform(y_data))
        train_labels.append(encoder.classes_)
    test_labels = []
    x_data_test = []
    y_data_test = []
    for x_data, y_data in zip(data["x_test"], data["y_test"]):
        x_data_test.append(x_data.astype('float32'))
        encoder.fit(y_data)
        y_data_test.append(encoder.transform(y_data))
        test_labels.append(encoder.classes_)

    return train_labels, test_labels, x_data_train, y_data_train, x_data_test, y_data_test


def preprocess_data_in_list_float32(data_list):
    for index, each_element in enumerate(data_list):
        data_list[index] = each_element.astype('float32')


def binarize_labels_in_list(labels_list):
    encoder = LabelBinarizer()
    all_y_data = np.concatenate(labels_list, axis=0)
    encoder.fit(all_y_data)
    for index, each_element in enumerate(labels_list):
        labels_list[index] = encoder.transform(each_element)

    return encoder.classes_

def binarize_labels_in_array(labels_array):
    encoder = LabelBinarizer()
    encoder.fit(labels_array)
    labels_array_t = encoder.transform(labels_array)

    return labels_array_t, encoder.classes_

def split_data_into_windows(x_data, y_data, window_size, step):
    """Splits sequence data into windows. Window size must be odd.

        # Arguments
            x_data:
            y_data:
            window_size:
            step:

        # Returns
            numpy.ndarrays: x_data_windowed, y_data_windowed
        # Raises

        """
    assert x_data.shape[0] == y_data.shape[0]
    assert window_size % 2 != 0
    N = x_data.shape[0]
    window_center_pos = math.floor(window_size / 2)
    half_window = math.floor(window_size / 2)
    x_data_windowed = []
    y_data_windowed = []
    while window_center_pos < (N - round(window_size / 2)):
        x_data_windowed.append(x_data[window_center_pos-half_window:window_center_pos + half_window + 1, :])
        y_data_windowed.append(y_data[window_center_pos, :])
        window_center_pos += step

    return np.asarray(x_data_windowed), np.asarray(y_data_windowed)

def checkEqual(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)

def transform_data_for_lstm(x_data, y_data):
    """Transforms the complete sequence for lstm input. Does not split data into windows

        # Arguments
            x_data:
            y_data:


        # Returns
            numpy.ndarrays: x_data_windowed, y_data_windowed
        # Raises

        """
    assert x_data.shape[0] == y_data.shape[0]
    # labels for all frames (=whole video) must be the same => one label per video
    assert checkEqual(y_data.tolist())

    x_data_windowed = []
    y_data_windowed = []

    x_data_windowed.append(x_data)
    y_data_windowed.append(y_data[0])

    return np.asarray(x_data_windowed), np.asarray(y_data_windowed)


def percentage_split_of_data_in_list(x_data, y_data, percentage_train):
    """Splits each element in the input lists into a training and testing set according to a percentage,
    then concatenates each list to return a numpy array.

        # Arguments
            x_data: list of features e.g. for each video
            y_data: list of labels e.g. vor each video
            percentage_train: precentage to keep for training

        # Returns
            numpy.ndarrays: x_train, y_train, x_test, y_test
        # Raises

        """
    assert len(x_data) == len(y_data)
    x_train, y_train, x_test, y_test = [], [], [], []
    for each_element_x, each_element_y in zip(x_data, y_data):
        split_slice = round(each_element_x.shape[0] * percentage_train / 100)
        x_train.append(each_element_x[:split_slice, :])
        x_test.append(each_element_x[split_slice:, :])
        y_train.append(each_element_y[:split_slice, :])
        y_test.append(each_element_y[split_slice:, :])

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    return x_train, y_train, x_test, y_test
    
    
    