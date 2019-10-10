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

def make_dense_autoencoder(in_shape, encoding_size=None):

    # Define network
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=in_shape),
        keras.layers.Dense(encoding_size, activation=tf.nn.relu),
        keras.layers.Dense(np.prod(in_shape), activation=tf.nn.sigmoid),
        keras.layers.Reshape(in_shape)
        ])
    return model

### end make_model

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

