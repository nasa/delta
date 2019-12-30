"""
Autoencoder.
"""
import os

import functools
import matplotlib.pyplot as plt
import numpy as np

import mlflow
import tensorflow as tf

from delta.config import config
from delta.imagery import imagery_dataset
from delta.ml.train import train
from delta.ml.networks import make_autoencoder


# TODO: Move this function!
def get_debug_bands(image_type):
    '''Pick the best bands to use for debug images'''
    bands = [0]
    if image_type == 'worldview':
        # TODO: Distinguish between WV2 and WV3
        bands = [4,2,1] # RGB
    # TODO: Support more sensors
    return bands

# With TF 1.12, the dataset needs to be constructed inside a function passed in to
# the estimator "train_and_evaluate" function to avoid getting a graph error!
def assemble_dataset():

    # Use wrapper class to create a Tensorflow Dataset object.
    # - The dataset will provide image chunks and corresponding labels.
    ids = imagery_dataset.AutoencoderDataset(config.dataset(), config.chunk_size(), config.chunk_stride())
    ds = ids.dataset()
    ds = ds.repeat(config.num_epochs()).batch(config.batch_size())
    ds = ds.prefetch(None)

    return ds


def assemble_dataset_for_predict():
    # Slightly simpler version of the previous function
    ids = imagery_dataset.AutoencoderDataset(config.dataset(), config.chunk_size(), config.chunk_stride())
    ds  = ids.dataset().batch(1) # Batch needed to match the original format
    return ds

def setup_parser(subparsers):
    sub = subparsers.add_parser('autoencode', help='Train an autoencder.')
    sub.add_argument("--num-debug-images", dest="num_debug_images", default=30, type=int,
                     help="Run this many images through the AE after training and write the "
                     "input/output pairs to disk.")
    sub.add_argument("--model", dest="model", required=True,
                     help="Location to save the model.")
    sub.set_defaults(function=main)
    config.setup_arg_parser(sub, labels=False, train=True)

def main(options):

    if not os.path.exists(config.output_folder()):
        os.mkdir(config.output_folder())

    config_d = config.dataset()
    aeds = imagery_dataset.AutoencoderDataset(config_d, config.chunk_size(), config.chunk_stride())

    print('Creating experiment')
    mlflow.log_param('image type',   config_d.image_type())
    mlflow.log_param('image folder', config_d.data_directory())
    mlflow.log_param('chunk size',   config.chunk_size())
    print('Creating model')
    data_shape = (aeds.chunk_size(), aeds.chunk_size(), aeds.num_bands())

    print('Training')
    # Estimator interface requires the dataset to be constructed within a function.
#     tf.logging.set_verbosity(tf.logging.INFO) # TODO 2.0

    # To do distribution of training with TF2/Keras, we need to create the model
    # in the scope of the distribution strategy (occurrs in the training function)
    model_fn = functools.partial(make_autoencoder,
                                 data_shape,
                                 encoding_size=int(config.num_hidden())
                                 )

    model, _ = train(model_fn, assemble_dataset(), config.training(), config.experiment_name())


    print('Saving Model')
    out_filename = os.path.join(config.output_folder(), options.model)
    tf.keras.models.save_model(model, out_filename, overwrite=True, include_optimizer = True)
    mlflow.log_artifact(out_filename)

    # Write input/output image pairs to the current working folder.
    print('Recording ', str(options.num_debug_images), ' demo images.')


    # Make a non-shuffled dataset with a simple iterator
    ds = assemble_dataset_for_predict()
    iterator = iter(ds)

    #scale = aeds.scale_factor()
    scale = 1.0
    num_bands = data_shape[-1]
#     debug_bands = get_debug_bands(config_values['input_dataset']['image_type'])
#     print('debug_bands = ' + str(debug_bands))
    for i in range(0, options.num_debug_images):
        print('Preparing debug image ' + str(i))

        value = next(iterator)
        element = model.predict(value)

        # Get the output value out of its weird format, then convert for image output
        pic = (element['reshape'][:, :, :] * scale).astype(np.uint8)
        pic = np.moveaxis(pic, 0, -1)

        in_pic = (value[0][0,:,:,:] * scale).astype(np.uint8)
        in_pic = np.moveaxis(in_pic, 0, -1)

        for band in range(num_bands):
            plt.subplot(num_bands,2,2*band+1)
            plt.imshow(in_pic[:,:,band])
            if band == 0:
                plt.title('Input image %03d' % (i, ))

            plt.subplot(num_bands,2,2*band+2)
            plt.imshow(pic[:,:,band])
            if band == 0:
                plt.title('Output image %03d' % (i, ))

        debug_image_filename = os.path.join(config.output_folder(),
                                            'Autoencoder_input_output_%03d.png' % (i, ))
        plt.savefig(debug_image_filename)

        mlflow.log_artifact(debug_image_filename)

    return 0
