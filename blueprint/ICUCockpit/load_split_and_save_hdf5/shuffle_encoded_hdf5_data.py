import random
import numpy as np
import h5py
import argparse
from dataset.config_reader import ConfigReader
import os 
from os import listdir
from os.path import isfile, join

class DataShuffler():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Shuffle multiple videos')
        parser.add_argument('-c', '--config_file', required=True, help='Configuration file with all the parameters')
        parser.add_argument('-i', '--input_directory', required=True, help='Input directory')
        self.args = parser.parse_args()
         
        self.config_reader = ConfigReader()
        self.config_reader.read(self.args.config_file)
        
        self.args = parser.parse_args()
        self.input_directory = self.args.input_directory
        self.file_pattern = self.config_reader.get("file_pattern")
        self.input_file_list = self.get_input_file_list() 
        self.window_size = self.config_reader.get("window_size")
        self.step = self.config_reader.get("step_size")
        self.label_list = self.config_reader.get("labels")
        self.output_file = self.args.input_directory + "/" + self.config_reader.get("output_file")
        print(" self.output_file",  self.output_file)
        self.xyt_data = []

    def get_input_file_list(self):
        return [join(self.input_directory,f) for f in listdir(self.input_directory) if isfile(join(self.input_directory, f)) and self.file_pattern in f]
       
    def generate_split_x_data_into_windows(self, x_data, date_time_data, window_size, step):
        """Splits sequence data into windows. Window size must be odd."""
        N = x_data.shape[0]
        window_start_pos = 0
        while window_start_pos < (N - window_size):
            yield (
                x_data[window_start_pos:window_start_pos + window_size, :],
                date_time_data[window_start_pos:window_start_pos + window_size, ]
            )
            window_start_pos += step

    def rreplace(self, s, old, new, occurrence):
        li = s.rsplit(old, occurrence)
        return new.join(li)

    def shuffle_data(self):
        x_data = []
        y_data = []
        date_time_data = []
        for file_name in self.input_file_list:
            print("file_name", file_name)
            f_in = h5py.File(file_name, 'r')
            for group_name, group_object in f_in.items():
                if group_name not in self.label_list:
                    continue
                print("group_name = ", group_name)
                for dataset_name, dataset_object in group_object.items():
                    print("dataset_name = ", dataset_name,"*", dataset_object)
                    if("_x" in dataset_name):
                        time_dataset_name = self.rreplace(dataset_name, "x", "time", 1)
                        my_generator = self.generate_split_x_data_into_windows(
                            dataset_object,
                            group_object[time_dataset_name],
                            self.window_size,
                            self.step
                        )
                        for x_item, date_time in my_generator:
                            x_data.append(x_item)
                            y_data.append(np.string_(group_name))
                            date_time_data.append(date_time)
            f_in.close()
        self.xyt_data = list(zip(x_data, y_data, date_time_data))
        random.shuffle(self.xyt_data)

    def save_data(self):
        x_data, y_data, date_time_data = zip(*self.xyt_data)
        hdf5_file = h5py.File(self.output_file, mode='w')
        x_data_shape = (len(y_data), len(x_data[0]), len(x_data[0][0]))
        hdf5_file.create_dataset("x_data", x_data_shape, np.float64 , data=x_data)
        hdf5_file.create_dataset("y_data", data=y_data)
        hdf5_file.create_dataset("date_and_time", data=date_time_data)

        hdf5_file.close()

if __name__ == '__main__':
    ds = DataShuffler()
    ds.shuffle_data()
    ds.save_data()

