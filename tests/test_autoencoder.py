from delta.ml import networks, train
import tensorflow as tf
from tensorflow import keras
import functools
import numpy as np

MNIST_WIDTH = 28 # The images are 28x28 pixels, single channel
MNIST_BANDS = 1
MNIST_MAX = 255.0 # Input images are 0-255

def dataset_fashion_mnist(batch_size=200, num_epochs=5, shuffle_buffer_size=10):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / MNIST_MAX
    (num_data, rows, cols) = train_images.shape
    train_images = np.reshape(train_images, (num_data, MNIST_WIDTH, MNIST_WIDTH, MNIST_BANDS))

    d_s = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(train_images),
                               tf.data.Dataset.from_tensor_slices(train_images)))
    d_s = d_s.shuffle(shuffle_buffer_size).repeat(num_epochs).batch(batch_size)
    return d_s


def test_dense_autoencoder():

    num_epochs = 1
    in_shape = (MNIST_WIDTH, MNIST_WIDTH, MNIST_BANDS)
    model_fn = functools.partial(networks.make_autoencoder, 
                                 in_shape, encoding_size=32, encoder_type='dense')
    data_fn = functools.partial(dataset_fashion_mnist, num_epochs=num_epochs)
   
    model, history = train.train(model_fn, dataset_fashion_mnist, num_epochs=num_epochs)
    assert history.history['loss'][-1] < 0.04, "Terminal Loss was greater than 0.04"

def test_conv_autoencoder():

    num_epochs = 1
    in_shape = (MNIST_WIDTH, MNIST_WIDTH, MNIST_BANDS)
    model_fn = functools.partial(networks.make_autoencoder, 
                                 in_shape, encoding_size=32, encoder_type='conv')
    data_fn = functools.partial(dataset_fashion_mnist, num_epochs=num_epochs)
   
    model, history = train.train(model_fn, dataset_fashion_mnist, num_epochs=num_epochs)
    assert history.history['loss'][-1] < 0.04, "Terminal Loss was greater than 0.04"

