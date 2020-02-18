import math
import h5py
import numpy as np
import collections
from utils import Stage
from utils import TestType
class DataGenerator():
    def __init__(self, input_file, batch_size, batch_type, epochs, train_labels, test_labels, test_type = None):
        self.input_file = input_file
        self.batch_size = batch_size
        self.batch_type = batch_type
        self.epochs = epochs
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.test_type = test_type

        self.train_start = 0
        self.train_end = 0.75
        self.validation_start = 0.75
        self.validation_end = 0.8
        self.test_start = 0.8
        self.test_end = 1

    def create_batches_mock(self):
        if self.batch_type == Stage.train:
            for i in range(50):
                data = np.random.randint(0, 100, size=(16, 100, 2048))
                y = np.zeros(16)
                yield data, y
        elif self.batch_type == Stage.validation:
            for i in range(10):
                data = np.random.randint(0, 100, size=(16, 100, 2048))
                y = np.zeros(16)
                yield data, y
        elif self.batch_type == Stage.test:
            for i in range(10):
                data = np.random.randint(0, 100, size=(16, 100, 2048))
                y = np.zeros(16)
                yield data, y

    def get_class_distribution(self):
        dictionary = collections.defaultdict(list)
        hdf5_file = h5py.File(self.input_file, mode='r')
        for i, n in enumerate(hdf5_file["y_data"]):
            dictionary[n.decode("utf-8")].append(i)

        return dictionary

    def generate_batches_for_one_epoch(self, batches_list):
        hdf5_file = h5py.File(self.input_file, mode='r')
        for n, value in enumerate(batches_list.items()):
            for i in value[1]:
                batch_x_data = []
                batch_y_data = []
                batch_date_time_data = []
                for j in range(self.batch_size):
                    batch_x_data.append(hdf5_file["x_data"][i, ...])
                    batch_y_data.append(hdf5_file["y_data"][i])
                    batch_date_time_data.append(hdf5_file["date_and_time"][i])

                yield batch_x_data, batch_y_data, batch_date_time_data

        hdf5_file.close()

    def generate_batches(self):
        hdf5_file = h5py.File(self.input_file, mode='r')
        batches_list = self.generate_list_of_batches_based_on_stage()
        if self.batch_type == Stage.train:
            for e in range(self.epochs):
                for n, value in enumerate(batches_list.items()):
                    for i in value[1][0]:
                        batch_x_data = []
                        batch_y_data = []
                        batch_date_time_data = []
                        for j in range(self.batch_size):
                            batch_x_data.append(hdf5_file["x_data"][i, ...])
                            batch_y_data.append(hdf5_file["y_data"][i])
                            batch_date_time_data.append(hdf5_file["date_and_time"][i])
                        yield batch_x_data, batch_y_data, batch_date_time_data
        else:
            for n, value in enumerate(batches_list.items()):
                for i in value[1][0]:
                    batch_x_data = []
                    batch_y_data = []
                    batch_date_time_data = []
                    for j in range(self.batch_size):
                        batch_x_data.append(hdf5_file["x_data"][i, ...])
                        batch_y_data.append(hdf5_file["y_data"][i])
                        batch_date_time_data.append(hdf5_file["date_and_time"][i])
                    yield batch_x_data, batch_y_data, batch_date_time_data
        hdf5_file.close()

    def generate_list_of_batches_based_on_stage(self):
        class_dictionary = self.get_class_distribution()
        batches_list = collections.defaultdict(list)

        if self.batch_type == Stage.train:
            for item in self.train_labels:
                for key, value in class_dictionary.items():
                    if item == key:
                       batches_list[item].append(value[self.train_start: math.ceil(self.train_end * len(value))])

        elif self.batch_type == Stage.validation:
            for item in self.train_labels:
                for key, value in class_dictionary.items():
                    if item == key:
                        batches_list[item].append(
                            value[
                                math.ceil(self.validation_start * len(value)):
                                math.ceil(self.validation_end * len(value))
                            ]
                        )
        elif self.batch_type == Stage.test:
            if self.test_type == TestType.positive:
                for item in self.train_labels:
                    for key, value in class_dictionary.items():
                        if item == key:
                            batches_list[item].append(
                                value[math.ceil(self.test_start * len(value)): math.ceil(self.test_end * len(value))])
            else:
                for item in self.test_labels:
                        for key, value in class_dictionary.items():
                            if item == key:
                                batches_list[item].append(value)

        else:
            print("Incorrect batch_type: train, validation or test")
            exit()
        return batches_list

    def len(self):
        # Denotes the number of batches per epoch
        hdf5_file = h5py.File(self.input_file, mode='r')
        data_length = hdf5_file["y_data"].shape[0]
        number_of_batches = int(math.ceil(float(data_length) / self.batch_size))
        if self.batch_type == Stage.train:
            return math.ceil(0.75 * number_of_batches)
        elif self.batch_type == Stage.validation:
            return math.ceil(0.05 * number_of_batches)
        elif self.batch_type == Stage.test:
            return math.ceil(0.2 * number_of_batches)
        else:
            print("Incorrect batch_type: train, validation or test")
            exit()

