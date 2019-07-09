"""
Script test out the image chunk generation calls.
"""
import sys
import os
import argparse
import random
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

from delta.imagery import landsat_utils #pylint: disable=C0413
from delta.imagery import worldview_utils #pylint: disable=C0413,W0611
from delta.imagery import imagery_dataset #pylint: disable=C0413
from delta.ml.train import Experiment


#------------------------------------------------------------------------------
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


def main(argsIn):

    usage  = "usage: train_autoencoder [options]"
    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument("--input-folder", dest="input_folder", required=True,
                        help="A folder with images for training the autoencoder.")

    parser.add_argument("--image-type", dest="image_type", required=True,
                        help="The image type (landsat, worldview, etc.).")

    parser.add_argument("--output-folder", dest="output_folder", required=True,
                        help="Write output chunk files to this folder.")

    parser.add_argument("--output-band", dest="output_band", type=int, default=0,
                        help="Only chunks from this band are written to disk.")

    parser.add_argument("--num-threads", dest="num_threads", type=int, default=1,
                        help="Number of threads to use for parallel image loading.")

    # Note: changed the default chunk size to 28.  Smaller chunks work better for
    # the toy network defined above.
    parser.add_argument("--chunk-size", dest="chunk_size", type=int, default=28,
                        help="The length of each side of the output image chunks.")

    parser.add_argument("--chunk-overlap", dest="chunk_overlap", type=int, default=0,
                        help="The amount of overlap of the image chunks.")

    parser.add_argument("--num-epochs", dest="num_epochs", type=int, default=70,
                        help="The number of epochs to train for")

    parser.add_argument("--num-hidden", dest="num_hidden", type=int, default=32,
                        help="The number of hidden elements in the autoencoder")

    parser.add_argument("--model-dest-name", dest="model_dest_name", type=str, default='autoencoder_model.keras.tf',
                        help="The name of the file where autoencoder is stored")

    try:
        options = parser.parse_args(argsIn)
    except argparse.ArgumentError:
        print(usage)
        return -1

    if not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)

    if options.image_type == 'landsat':
        num_bands = len(landsat_utils.get_landsat_bands_to_use('LS8'))
    elif options.image_type == 'worldview':
        num_bands = len(worldview_utils.get_worldview_bands_to_use('WV02'))
    elif options.image_type == 'rgba':
        num_bands = 3
    else:
        print('Unsupported image type %s.' % (options.image_type))
    ### end if

    print('loading data from ' + options.input_folder)
    ids = imagery_dataset.AutoencoderDataset(options.image_type, image_folder=options.input_folder, chunk_size=options.chunk_size)
    ds = ids.dataset()

    BATCH_SIZE = 2
    num_entries = ids.total_num_regions()
    if num_entries < BATCH_SIZE:
        raise Exception('BATCH_SIZE (%d) is too large for the number of input regions (%d)!' 
                % (BATCH_SIZE, num_entries))
    ### end if
    ds = ds.batch(BATCH_SIZE)
    ds = ds.repeat()

    # TF additions
    # If the mlfow directory doesn't exist, create it.
    mlflow_tracking_dir = os.path.join(options.output_folder,'mlruns')
    if not os.path.exists(mlflow_tracking_dir):
        os.mkdir(mlflow_tracking_dir)
    ### end if

    print('Creating experiment')
    experiment = Experiment(mlflow_tracking_dir, 'autoencoder_%s'%(options.image_type,), output_dir=options.output_folder)
    mlflow.log_param('image type', options.image_type)
    mlflow.log_param('image folder', options.input_folder)
    mlflow.log_param('chunk size', options.chunk_size)
    print('Creating model')
    model = make_model(ids.data_shape(), encoding_size=options.num_hidden)
    print('Training')
    experiment.train(model, ds, steps_per_epoch=ids.steps_per_epoch(), batch_size=BATCH_SIZE)

    print('Saving Model')
    if options.model_dest_name is not None:
        out_filename = os.path.join(options.output_folder,options.model_dest_name)
        tf.keras.models.save_model(model,out_filename,overwrite=True,include_optimizer=True)
        mlflow.log_artifact(out_filename)
    ### end if
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
