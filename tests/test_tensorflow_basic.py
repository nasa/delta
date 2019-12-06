"""
file: test_tensorflow_basic.py
purpose: A file which tests running tensorflow on the MNIST dataset.
date: 2019.03.18
"""
import tensorflow as tf
from delta.ml import train

def test_find_cpus():
    '''
    Tests that tensorflow can find the available CPUs.
    '''
    assert train.get_devices(0), "Could not find any CPU Logical Devices"

# disable this tests so tests can pass without gpu
#def test_find_gpus():
#    '''
#    Tests that tensorflow can find the available GPUs.
#    Note: this will fail if tested on a computer without a GPU that Tensorflow
#    recognizes.
#    '''
#    assert len(train.get_devices(1)) > 0, "Could not find any GPU Logical Devices"


def test_mnist_train():
    """ Tests the tensorflow library on the MNIST data set. """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, verbose=0)
    (_, accuracy) = model.evaluate(x_test, y_test, verbose=0)
    assert accuracy > 0.96
