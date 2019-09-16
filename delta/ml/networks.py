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
        keras.layers.Dense(np.prod(in_shape), activation=tf.nn.sigmoid),
        keras.layers.Reshape(in_shape)
        ])
    print(model.summary())
    return model

### end make_model
