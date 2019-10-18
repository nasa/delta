import tensorflow as tf
from tensorflow import keras
import mlflow
import numpy as np

def make_autoencoder(in_shape, encoding_size=32,encoder_type='conv'):
    assert isinstance(encoding_size, int)

    mlflow.log_param('input_size',str(in_shape))
    mlflow.log_param('encoding_size',encoding_size)

    model = None
    if encoder_type == 'conv':
        model = make_convolutional_autoencoder(in_shape, encoding_size=encoding_size)
    elif encoder_type == 'dense':
        model = make_dense_autoencoder(in_shape, encoding_size=encoding_size)
    else:
        print('ERROR: Unrecognized autoencoder type %s' % (encoder_type,))
        exit()
    ### end if
    return model
### end make_autoencoder

def make_dense_autoencoder(in_shape, encoding_size=None):

    # Define network
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=in_shape),
        keras.layers.Dense(encoding_size, activation=tf.nn.relu),
        keras.layers.Dense(np.prod(in_shape), activation=tf.nn.relu),
        keras.layers.Reshape(in_shape)
        ])
    return model
### end make_dense_autoencoder

def make_convolutional_autoencoder(in_shape, encoding_size=None):

    model = keras.models.Sequential([
        # Encoder
        keras.layers.InputLayer(input_shape=in_shape),
        keras.layers.Conv2D(encoding_size, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(encoding_size*2, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(encoding_size*4, (3, 3), activation='relu', padding='same'),
        # Decoder
        keras.layers.Conv2D(encoding_size*4, (3, 3), activation='relu', padding='same'),
        keras.layers.UpSampling2D((2, 2)),
        keras.layers.Conv2D(encoding_size*2, (3, 3), activation='relu', padding='same'),
        keras.layers.UpSampling2D((2, 2)),
        keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])
    return model
### end make_convolutional_autoencoder

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
