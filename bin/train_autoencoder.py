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

from delta import config #pylint: disable=C0413
from delta.imagery.sources import landsat #pylint: disable=C0413
from delta.imagery.sources import worldview #pylint: disable=C0413,W0611
from delta.imagery import imagery_dataset #pylint: disable=C0413
from delta.ml.train import Experiment

#------------------------------------------------------------------------------
def make_model(in_shape, encoding_size=32):

    mlflow.log_param('input_size',str(in_shape))
    mlflow.log_param('encoding_size',encoding_size)
    batch_in_shape = in_shape
    print( 'Input shape', in_shape)

    # Define network
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=batch_in_shape, data_format='channels_first'),
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

    parser.add_argument("--config-file", dest="config_file", required=False,
                        help="Dataset configuration file.")

    parser.add_argument("--data-folder", dest="data_folder", required=False,
                        help="Specify data folder instead of supplying config file.")
    parser.add_argument("--image-type", dest="image_type", required=False,
                        help="Specify image type along with the data folder. (landsat, worldview, or rgba)")
    parser.add_argument("--label-folder", dest="label_folder", required=False,
                        help="Specify label folder instead of supplying config file.")

    parser.add_argument("--output-folder", dest="output_folder", required=True,
                        help="Write output chunk files to this folder.")

    parser.add_argument("--model-dest", dest="model_dest_name", required=False, default=False,
                        help="Name of the file for storing the learned model.")

    parser.add_argument("--test-limit", dest="test_limit", type=int, default=0,
                        help="If set, use a maximum of this many input values for training.")

    try:
        options = parser.parse_args(argsIn)
    except argparse.ArgumentError:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if not options.config_file and not (options.data_folder and options.image_type):
        parser.print_help(sys.stderr)
        print('Must specify either --config-file or --data-folder and --image-type')
        sys.exit(1)

    if options.output_folder is not None  and not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)

    config_values = config.parse_config_file(options.config_file,
                                             options.data_folder, options.image_type)

    # Because we are doing an autoencoder, the labels are the data.
    config_values['input_dataset']['label_directory'] = options.data_folder
    config_values['input_dataset']['label_postfix'] = ''
    if 'num_hidden' not in config_values['ml']:
        config_values['ml']['num_hidden'] = 32
    batch_size = config_values['ml']['batch_size']
    num_epochs = config_values['ml']['num_epochs']

    if options.image_type == 'landsat':
        num_bands = len(landsat.get_landsat_bands_to_use('LS8'))
    elif options.image_type == 'worldview':
        num_bands = len(worldview.get_worldview_bands_to_use('WV02'))
    elif options.image_type == 'rgba':
        num_bands = 3
    else:
        print('Unsupported image type %s.' % (options.image_type))
    ### end if

    print('loading data from ' + options.data_folder)
    ids = imagery_dataset.AutoencoderDataset(config_values)
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
    for k in config_values.keys():
        experiment.log_parameters(config_values[k])
#     mlflow.log_param('image type', options.image_type)
#     mlflow.log_param('image folder', options.input_folder)
#     mlflow.log_param('chunk size', options.chunk_size)
    print('Creating model')
    model = make_model(ids.data_shape(), encoding_size=config_values['ml']['num_hidden'])
    print('Training')
#     experiment.train(model, ds, steps_per_epoch=ids.steps_per_epoch())
    experiment.train(model, ds, steps_per_epoch=578)

    print('Saving Model')
    if options.model_dest_name is not None:
        out_filename = os.path.join(options.output_folder,options.model_dest_name)
        tf.keras.models.save_model(model,out_filename,overwrite=True,include_optimizer=True)
        mlflow.log_artifact(out_filename)
    ### end if
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
