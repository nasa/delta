"""
Script test out the image chunk generation calls.
"""
import sys
import os
import argparse
import functools
#import random
import numpy as np

### Tensorflow includes

import mlflow

import tensorflow as tf
from tensorflow import keras
#import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TODO: Make sure this goes everywhere!
if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

from delta import config #pylint: disable=C0413
from delta.ml.train import Experiment  #pylint: disable=C0413


MNIST_WIDTH = 28 # The images are 28x28 pixels, single channel
MNIST_BANDS = 1
MNIST_MAX = 255.0 # Input images are 0-255


#------------------------------------------------------------------------------

# TODO: Move this function to a shared location!!!
# Use the same model creation function as our tool
def make_model(in_shape, encoding_size=32):

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


# With TF 1.12, the dataset needs to be constructed inside a function passed in to
# the estimator "train_and_evaluate" function to avoid getting a graph error!
def assemble_mnist_dataset(batch_size, num_epochs=1, shuffle_buffer_size=1000,
                           use_fraction=1.0, get_test=False):

    print("Loading Fashion-MNIST")
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    amount_train = int(train_images.shape[0] * use_fraction)
    amount_test  = int(test_images.shape[0]  * use_fraction)
    train_images = train_images[:amount_train] / MNIST_MAX
    test_images  = test_images[:amount_test]   / MNIST_MAX
    train_labels = train_labels[:amount_train]
    test_labels  = test_labels[:amount_test]
    print('Num images loaded: train=', amount_train, ' test=', amount_test)
    train_images = np.reshape(train_images, (amount_train, MNIST_WIDTH, MNIST_WIDTH, MNIST_BANDS))
    test_images  = np.reshape(test_images,  (amount_test,  MNIST_WIDTH, MNIST_WIDTH, MNIST_BANDS))

    # Return the selected dataset
    # - Since it is the autoencoder test, the labels are the same as the input images
    if get_test:
        ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(test_images),
                                  tf.data.Dataset.from_tensor_slices(test_images)))
    else:
        ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(train_images),
                                  tf.data.Dataset.from_tensor_slices(train_images)))

    # TODO: Check this!
    ds = ds.shuffle(shuffle_buffer_size).repeat(num_epochs).batch(batch_size)

    return ds

def main(argsIn):

    usage  = "usage: train_autoencoder [options]"
    parser = argparse.ArgumentParser(usage=usage)

    parser = argparse.ArgumentParser(usage='train_autoencoder.py [options]')

    parser.add_argument("--config-file", dest="config_file", required=True,
                        help="Dataset configuration file.")

    parser.add_argument("--use-fraction", dest="use_fraction", default=1.0, type=float,
                        help="Only use this fraction of the MNIST data, to reduce processing time.")

    parser.add_argument("--shuffle-buffer-size", dest="shuffle_buffer_size", default=2000, type=int,
                        help="Size of the input shuffle buffer.")

    parser.add_argument("--num-gpus", dest="num_gpus", default=0, type=int,
                        help="Try to use this many GPUs.")

    try:
        options = parser.parse_args(argsIn)
    except argparse.ArgumentError:
        print(usage)
        return -1

    config_values = config.parse_config_file(options.config_file, None, None, no_required=False)

    batch_size = config_values['ml']['batch_size']
    num_epochs = config_values['ml']['num_epochs']

    config_values['ml']['chunk_size'] = MNIST_WIDTH

    output_folder = config_values['ml']['output_folder']
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


    # Get the MNIST train and test datasets to pass to the estimator
    dataset_train_fn = functools.partial(assemble_mnist_dataset, batch_size,
                                         num_epochs, options.shuffle_buffer_size,
                                         options.use_fraction, get_test=False)
    dataset_test_fn  = functools.partial(assemble_mnist_dataset, batch_size,
                                         num_epochs, options.shuffle_buffer_size,
                                         options.use_fraction, get_test=True)


    # If the mlfow directory doesn't exist, create it.
    mlflow_tracking_dir = os.path.join(output_folder,'mlruns')
    if not os.path.exists(mlflow_tracking_dir):
        os.mkdir(mlflow_tracking_dir)
    ### end if

    print('Creating experiment')
    experiment = Experiment(mlflow_tracking_dir, 'autoencoder_MNIST', output_dir=output_folder)
    print('Creating model')
    data_shape = (MNIST_BANDS, MNIST_WIDTH, MNIST_WIDTH)
    model = make_model(data_shape, encoding_size=config_values['ml']['num_hidden'])
    print('Training')

#     experiment.train(model, ds, steps_per_epoch=1000, batch_size=batch_size)
    #experiment.train(model, ds, steps_per_epoch=1000,log_model=False)

    # Estimator interface requires the dataset to be constructed within a function.
    tf.logging.set_verbosity(tf.logging.INFO)
    experiment.train_estimator(model, dataset_train_fn, dataset_test_fn,
                               steps_per_epoch=1000, log_model=False,
                               num_gpus=options.num_gpus)

    print('Saving Model')
    if config_values['ml']['model_dest_name'] is not None:
        out_filename = os.path.join(output_folder, config_values['ml']['model_dest_name'])
        tf.keras.models.save_model(model, out_filename, overwrite=True, include_optimizer=True)
        mlflow.log_artifact(out_filename)
    ### end if

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
