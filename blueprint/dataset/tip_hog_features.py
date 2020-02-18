

import numpy as np
import tensorflow as tf
import random as rn
import os

def load_data():
    """Loads the IMDB dataset.

    # Arguments


    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    # Raises
       

    """
    # load my data
input_file = args.input

# comma delimited is the default
print("Loading data...")
df = pd.read_csv(input_file, delimiter = ",", header = None)
#print("Data types in Data Frame: ", df.dtypes)

# get numeric data only (remove the non-numeric columns)
df_num = df._get_numeric_data()

# get label data
df_str = df.select_dtypes(include=['object'])
y = df_str.values
print("labels.shape: ", y.shape)
#print("labels: ", labels)

# create a numpy array with the numeric values for input into scikit-learn
data = df_num.as_matrix()

# select columns
X = data[:, 3:]  # select columns 3 through end
meta_data = data[:, :3]   # select column 0-2, additional info (frame etc.)

print("X: ", X.shape)
print("meta_data: ", meta_data.shape)
print("y: ", y.shape)


