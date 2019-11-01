"""
Script test out the image chunk generation calls.
"""
import sys
import os
import os.path
import argparse
import functools
import matplotlib.pyplot as plt
import numpy as np

import mlflow

import tensorflow as tf
from tensorflow import keras

from delta.config import config
from delta.ml.train import Experiment
from delta.ml.networks import make_autoencoder

MNIST_WIDTH = 28 # The images are 28x28 pixels, single channel
MNIST_BANDS = 1
MNIST_MAX = 255.0 # Input images are 0-255


#------------------------------------------------------------------------------


# With TF 1.12, the dataset needs to be constructed inside a function passed in to
# the estimator "train_and_evaluate" function to avoid getting a graph error!
def assemble_mnist_dataset(batch_size, num_epochs=1, shuffle_buffer_size=1000,
                           use_fraction=1.0, get_test=False):
    """Loads the mnist handwritten digits dataset"""

    print("Loading Fashion-MNIST")
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    amount_train = int(train_images.shape[0] * use_fraction)
    amount_test = int(test_images.shape[0]  * use_fraction)
    train_images = train_images[:amount_train] / MNIST_MAX
    test_images = test_images[:amount_test]   / MNIST_MAX
    train_labels = train_labels[:amount_train]
    test_labels = test_labels[:amount_test]
    print('Num images loaded: train=', amount_train, ' test=', amount_test)
    train_images = np.reshape(train_images, (amount_train, MNIST_WIDTH, MNIST_WIDTH, MNIST_BANDS))
    test_images = np.reshape(test_images, (amount_test, MNIST_WIDTH, MNIST_WIDTH, MNIST_BANDS))

    # Return the selected dataset
    # - Since it is the autoencoder test, the labels are the same as the input images
    if get_test:
        d_s = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(test_images),
                                   tf.data.Dataset.from_tensor_slices(test_images)))
    else:
        d_s = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(train_images),
                                   tf.data.Dataset.from_tensor_slices(train_images)))

    d_s = d_s.shuffle(shuffle_buffer_size).repeat(num_epochs).batch(batch_size)

    return d_s

def assemble_mnist_dataset_for_predict(test_count, no_labels=False):
    """Loads the mnist fashion dataset just for prediction.
       If no_labels is set only the input image portion of the dataset is used"""

    fashion_mnist = keras.datasets.fashion_mnist
    (_, _), (test_images, _) = fashion_mnist.load_data()
    test_images = test_images[:test_count]   / MNIST_MAX
    test_images = np.reshape(test_images, (test_count, MNIST_WIDTH, MNIST_WIDTH, MNIST_BANDS))

    if no_labels:
        return tf.data.Dataset.from_tensor_slices(test_images)

    return tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(test_images),
                                tf.data.Dataset.from_tensor_slices(test_images)))


def main(args_in): #pylint: disable=R0914
    """Main function for executing MNIST training test"""

    parser = argparse.ArgumentParser(usage='train_autoencoder.py [options]')

    parser.add_argument("--use-fraction", dest="use_fraction", default=1.0, type=float,
                        help="Only use this fraction of the MNIST data, to reduce processing time.")

    parser.add_argument("--num-debug-images", dest="num_debug_images", default=0, type=int,
                        help="Run this many images through the AE after training and write the "
                        "input/output pairs to disk.")

    parser.add_argument("--load-model", action="store_true", dest="load_model", default=False,
                        help="Start with the model saved in the current output location.")
    options = config.parse_args(parser, args_in, labels=False)

    config.set_value('ml', 'chunk_size', MNIST_WIDTH)

    if not os.path.exists(config.output_folder()):
        os.mkdir(config.output_folder())


    # Get the MNIST train and test datasets to pass to the estimator
    dataset_train_fn = functools.partial(assemble_mnist_dataset, config.batch_size(), config.num_epochs(),
                                         config.dataset().shuffle_buffer_size(),
                                         options.use_fraction, get_test=False)
    #dataset_test_fn = functools.partial(assemble_mnist_dataset, batch_size, num_epochs,
    #                                    config_values['input_dataset']['shuffle_buffer_size'],
    #                                    options.use_fraction, get_test=True)

    # If the mlfow directory doesn't exist, create it.
    mlflow_tracking_dir = os.path.join(config.output_folder(), 'mlruns')
    if not os.path.exists(mlflow_tracking_dir):
        os.mkdir(mlflow_tracking_dir)
    ### end if

    print('Creating experiment')
    experiment = Experiment(mlflow_tracking_dir,
                            'autoencoder_Fashion_MNIST',
                            output_dir=config.output_folder())

    out_filename = os.path.join(config.output_folder(), config.model_dest_name())
    if options.load_model:
        print('Loading model from ' + out_filename)
        model_fn = functools.partial(tf.keras.models.load_model, out_filename)
    else:
        print('Creating model')
        data_shape = (MNIST_WIDTH, MNIST_WIDTH, MNIST_BANDS)
        #model = make_autoencoder(data_shape, encoding_size=config_values['ml']['num_hidden'])
        model_fn = functools.partial(make_autoencoder, data_shape,
                                     encoding_size=config.num_hidden())

    print('Training')
    # Estimator interface requires the dataset to be constructed within a function.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO) # TODO 2.0
    model, _ = experiment.train_keras(model_fn, dataset_train_fn,
                                      num_epochs=config.num_epochs(),
                                      num_gpus=config.num_gpus())

    print('Saving Model')
    if config.model_dest_name() is not None:
        model.save(out_filename, overwrite=True, include_optimizer=True)
        mlflow.log_artifact(out_filename)
    ### end if

    # Write input/output image pairs to the current working folder.
    print('Recording ', str(options.num_debug_images), ' demo images.')

    # Make a non-shuffled dataset with a simple iterator
    # - For some reason we can't have the label part when loaded from disk
    ds = assemble_mnist_dataset_for_predict(options.num_debug_images,
                                            no_labels=options.load_model)
    iterator = iter(ds.batch(1))

    for i in range(0, options.num_debug_images):

        # Get the next image pair and predict from it
        value = next(iterator)
        element = model.predict(value)

        # Get the output value out of its weird format, then convert for image output
        pic = (element[0,:, :, 0] * MNIST_MAX).astype(np.uint8)

        # For some reason the Keras model works differently if originally loaded from disk!
        if options.load_model:
            input_image = value[0,:,:,0]
        else:
            input_image = value[0][0,:,:,0]

        plt.subplot(1,2,1)
        plt.imshow(input_image)
        plt.title('Input image %03d' % (i, ))

        plt.subplot(1,2,2)
        plt.imshow(pic)
        plt.title('Output image %03d' % (i, ))

        debug_image_filename = os.path.join(config.output_folder(),
                                            'Fashion_MNIST_input_output_%03d.png' % (i, ))
        plt.savefig(debug_image_filename)

        mlflow.log_artifact(debug_image_filename)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
