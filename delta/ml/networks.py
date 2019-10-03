import tensorflow as tf
from tensorflow import keras
import mlflow
import numpy as np


# TODO: Move this function to a shared location!!!
# Use the same model creation function as our tool
def make_autoencoder(in_shape, encoding_size=32):

    mlflow.log_param('input_size',str(in_shape))
    mlflow.log_param('encoding_size',encoding_size)


    # Define network
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=in_shape),
        keras.layers.Dense(encoding_size, activation=tf.nn.relu),
        keras.layers.Dense(np.prod(in_shape), activation=tf.nn.relu),
        keras.layers.Reshape(in_shape)
        ])
    print(model.summary())
    return model

### end make_autoencoder

def make_task_specific(in_shape, out_shape):
    '''
    Constructs an arbitrarily sized dense neural network.
    TODO: Come up with a more principled network selection method.
    '''
    fc1_size = np.prod(in_shape)
    # There is no reason behind any of the following values.
    fc2_size = 253
    fc3_size = 253
    fc4_size = 81

    dropout_rate = 0.3

    mlflow.log_param('fc1_size',fc1_size)
    mlflow.log_param('fc2_size',fc2_size)
    mlflow.log_param('fc3_size',fc3_size)
    mlflow.log_param('fc4_size',fc4_size)
    mlflow.log_param('dropout_rate',dropout_rate)


    # Define network
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=in_shape),
        keras.layers.Dense(fc2_size, activation=tf.nn.relu),
        keras.layers.Dropout(rate=dropout_rate),
        keras.layers.Dense(fc3_size, activation=tf.nn.relu),
        keras.layers.Dropout(rate=dropout_rate),
        keras.layers.Dense(fc4_size, activation=tf.nn.relu),
        keras.layers.Dropout(rate=dropout_rate),
        keras.layers.Dense(out_shape, activation=tf.nn.sigmoid),
        keras.layers.Softmax()
        ])
    return model

### end make_task_specific
