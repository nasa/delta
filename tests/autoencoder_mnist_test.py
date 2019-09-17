"""
Script test out the image chunk generation calls.
"""
import sys
import os
import os.path
import argparse
import functools
#import random
import matplotlib.pyplot as plt
import numpy as np
#pylint: disable=C0413

### Tensorflow includes

import mlflow

import tensorflow as tf
from tensorflow import keras
#import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from delta import config #pylint: disable=C0413
from delta.ml.train import Experiment  #pylint: disable=C0413
from delta.ml.networks import make_autoencoder


MNIST_WIDTH = 28 # The images are 28x28 pixels, single channel
MNIST_BANDS = 1
MNIST_MAX = 255.0 # Input images are 0-255


#------------------------------------------------------------------------------


# With TF 1.12, the dataset needs to be constructed inside a function passed in to
# the estimator "train_and_evaluate" function to avoid getting a graph error!
def assemble_mnist_dataset(batch_size, num_epochs=1, shuffle_buffer_size=1000,
                           use_fraction=1.0, get_test=False):
    '''
    Loads the mnist handwritten digits dataset.
    '''

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

def assemble_mnist_dataset2(batch_size, test_count):
    '''
    Loads the mnist fashion dataset
    '''

    fashion_mnist = keras.datasets.fashion_mnist
    (_, _), (test_images, test_labels) = fashion_mnist.load_data()
    test_images = test_images[:test_count]   / MNIST_MAX
    test_labels = test_labels[:test_count]
    #test_images = np.reshape(test_images, (test_count, MNIST_WIDTH, MNIST_WIDTH, MNIST_BANDS))

    d_s = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(test_images),
                               tf.data.Dataset.from_tensor_slices(test_images)))

    d_s = d_s.batch(batch_size)

    return d_s


def main(args_in): #pylint: disable=R0914
    '''
    Main function for executing MNIST training test.
    '''

    usage = "usage: train_autoencoder [options]"
    parser = argparse.ArgumentParser(usage=usage)

    parser = argparse.ArgumentParser(usage='train_autoencoder.py [options]')

    parser.add_argument("--config-file", dest="config_file", required=True,
                        help="Dataset configuration file.")

    parser.add_argument("--use-fraction", dest="use_fraction", default=1.0, type=float,
                        help="Only use this fraction of the MNIST data, to reduce processing time.")

    parser.add_argument("--num-debug-images", dest="num_debug_images", default=0, type=int,
                        help="Run this many images through the AE after training and write the "
                        "input/output pairs to disk.")

    parser.add_argument("--num-gpus", dest="num_gpus", default=0, type=int,
                        help="Try to use this many GPUs.")

    try:
        options = parser.parse_args(args_in)
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
    dataset_train_fn = functools.partial(assemble_mnist_dataset, batch_size, num_epochs,
                                         config_values['input_dataset']['shuffle_buffer_size'],
                                         options.use_fraction, get_test=False)
    dataset_test_fn = functools.partial(assemble_mnist_dataset, batch_size, num_epochs,
                                        config_values['input_dataset']['shuffle_buffer_size'],
                                        options.use_fraction, get_test=True)

    # If the mlfow directory doesn't exist, create it.
    mlflow_tracking_dir = os.path.join(output_folder, 'mlruns')
    if not os.path.exists(mlflow_tracking_dir):
        os.mkdir(mlflow_tracking_dir)
    ### end if

    print('Creating experiment')
    experiment = Experiment(mlflow_tracking_dir, 'autoencoder_Fashion_MNIST', output_dir=output_folder)
    print('Creating model')
    data_shape = (MNIST_WIDTH, MNIST_WIDTH, MNIST_BANDS)
    model = make_autoencoder(data_shape, encoding_size=config_values['ml']['num_hidden'])
    print('Training')

#     experiment.train(model, ds, steps_per_epoch=1000, batch_size=batch_size)
    #experiment.train(model, ds, steps_per_epoch=1000,log_model=False)

    # Estimator interface requires the dataset to be constructed within a function.
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = experiment.train_estimator(model, dataset_train_fn, dataset_test_fn,
                                           steps_per_epoch=1000, log_model=False,
                                           num_gpus=options.num_gpus)

    print('Saving Model')
    if config_values['ml']['model_dest_name'] is not None:
        out_filename = os.path.join(output_folder, config_values['ml']['model_dest_name'])
        tf.keras.models.save_model(model, out_filename, overwrite=True, include_optimizer=True)
        mlflow.log_artifact(out_filename)
    ### end if

    # Write input/output image pairs to the current working folder.
    print('Recording ', str(options.num_debug_images), ' demo images.')

    output_ds1 = functools.partial(assemble_mnist_dataset2, batch_size, options.num_debug_images)
    (_, _), (test_images, _) = keras.datasets.fashion_mnist.load_data()
    result = estimator.predict(output_ds1)

    for i in range(0, options.num_debug_images):

        element = next(result)
        pic = (element['reshape'][:, :, 0] * MNIST_MAX).astype(np.uint8)

        plt.subplot(1,2,1)
        plt.imshow(test_images[i])
        plt.title('Input image %03d' % (i, ))

        plt.subplot(1,2,2)
        plt.imshow(pic)
        plt.title('Output image %03d' % (i, ))

        debug_image_filename = os.path.join(output_folder,
                                            'Fashion_MNIST_input_output_%03d.png' % (i, ))
        plt.savefig(debug_image_filename)

        mlflow.log_artifact(debug_image_filename)

#        plt.imsave(os.path.join(output_folder, 'input'+str(i)+'.png'), test_images[i])
#
#        element = next(result)
#        #print(element)
#        #print(element['reshape'].shape)
#        pic = (element['reshape'][:, :, 0] * MNIST_MAX).astype(np.uint8)
#        plt.imsave(os.path.join(output_folder, 'output'+str(i)+'.png'), pic)
#
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
