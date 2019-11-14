'''
file: test_autoencoder.py -- Runs the autoencoder from the delta project through some
        simple unit tests.
date: 2019.10.29
'''

import functools
import numpy as np

import tensorflow as tf
from tensorflow import keras
from delta.ml import networks, train

MNIST_WIDTH = 28 # The images are 28x28 pixels, single channel
MNIST_BANDS = 1
MNIST_MAX = 255.0 # Input images are 0-255

def dataset_fashion_mnist(batch_size=200, num_epochs=5, shuffle_buffer_size=10):
    '''
    Creates a Tensorflow dataset out of the MNIST fashion dataset for training an
    autoencode
    '''
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, _), (_, _) = fashion_mnist.load_data()
    train_images = train_images / MNIST_MAX
    (num_data, _, _) = train_images.shape
    reduce_by = 16
    train_images = train_images[:(num_data // reduce_by), :, :]
    train_images = np.reshape(train_images, (num_data // reduce_by, MNIST_WIDTH, MNIST_WIDTH, MNIST_BANDS))

    d_s = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(train_images),
                               tf.data.Dataset.from_tensor_slices(train_images)))
    d_s = d_s.shuffle(shuffle_buffer_size).batch(batch_size).repeat(num_epochs)
    return d_s


def test_dense_autoencoder():
    '''
    tests the performance of the dense autoencoder on the Fashion MNIST dataset
    '''

    num_epochs = 2
    in_shape = (MNIST_WIDTH, MNIST_WIDTH, MNIST_BANDS)
    model_fn = functools.partial(networks.make_autoencoder,
                                 in_shape, encoding_size=32, encoder_type='dense')
    data_fn = functools.partial(dataset_fashion_mnist, num_epochs=num_epochs)

    model, history = train.train(model_fn, data_fn, num_epochs=num_epochs)
    assert model is not None
    assert history.history['loss'][-1] < 0.1, "Terminal Loss was greater than 0.08"

def test_conv_autoencoder():
    '''
    tests the performance of the convolutional autoencoder on the Fashion MNIST dataset
    '''

    num_epochs = 1
    in_shape = (MNIST_WIDTH, MNIST_WIDTH, MNIST_BANDS)
    model_fn = functools.partial(networks.make_autoencoder,
                                 in_shape, encoding_size=32, encoder_type='conv')
    data_fn = functools.partial(dataset_fashion_mnist, num_epochs=num_epochs)

    model, history = train.train(model_fn, data_fn, num_epochs=num_epochs)
    assert model is not None
    assert history.history['loss'][-1] < 0.1, "Terminal Loss was greater than 0.08"
