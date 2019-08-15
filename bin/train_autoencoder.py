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

    parser = argparse.ArgumentParser(usage='sc_process_test.py [options]')

    parser.add_argument("--config-file", dest="config_file", required=False,
                        help="Dataset configuration file.")
    parser.add_argument("--data-folder", dest="data_folder", required=False,
                        help="Specify data folder instead of supplying config file.")
    parser.add_argument("--image-type", dest="image_type", required=False,
                        help="Specify image type along with the data folder. (landsat, landsat-simple, worldview, or rgba)")

    try:
        options = parser.parse_args(argsIn)
    except argparse.ArgumentError:
        print(usage)
        return -1

    config_values = config.parse_config_file(options.config_file,
                                             options.data_folder, options.image_type)

    batch_size = config_values['ml']['batch_size']
    num_epochs = config_values['ml']['num_epochs']


    if not os.path.exists(config_values['ml']['output_folder']):
        os.mkdir(options.output_folder)

    if options.image_type == 'landsat':
        num_bands = len(landsat.get_landsat_bands_to_use('LS8'))
    elif options.image_type == 'worldview':
        num_bands = len(worldview.get_worldview_bands_to_use('WV02'))
    elif options.image_type == 'rgba':
        num_bands = 3
    else:
        print('Unsupported image type %s.' % (options.image_type))
    ### end if

    print('loading data from ' + config_values['input_dataset']['data_directory'])
    aeds = imagery_dataset.AutoencoderDataset(config_values)
    ds = aeds.dataset()

#     num_entries = aeds.total_num_regions()
#     if num_entries < batch_size:
#         raise Exception('BATCH_SIZE (%d) is too large for the number of input regions (%d)!' 
#                 % (batch_size, num_entries))
    ### end if
    ds = ds.batch(batch_size)
    ds = ds.repeat()

    # TF additions
    # If the mlfow directory doesn't exist, create it.
    mlflow_tracking_dir = os.path.join(config_values['ml']['output_folder'],'mlruns')
    if not os.path.exists(mlflow_tracking_dir):
        os.mkdir(mlflow_tracking_dir)
    ### end if

    print('Creating experiment')
    experiment = Experiment(mlflow_tracking_dir, 'autoencoder_%s'%(options.image_type,), output_dir=config_values['ml']['output_folder'])
    mlflow.log_param('image type', options.image_type)
    mlflow.log_param('image folder', config_values['input_dataset']['data_directory'])
    mlflow.log_param('chunk size', config_values['ml']['chunk_size'])
    print('Creating model')
    data_shape = (aeds.num_bands(), aeds.chunk_size(), aeds.chunk_size())
    model = make_model(data_shape, encoding_size=config_values['ml']['num_hidden'])
    print('Training')
    
#     experiment.train(model, ds, steps_per_epoch=1000, batch_size=batch_size)
    experiment.train(model, ds, steps_per_epoch=1000,log_model=False)

    print('Saving Model')
    if options.model_dest_name is not None:
        out_filename = os.path.join(options.output_folder,config_values['ml']['model_dest_name'])
        tf.keras.models.save_model(model,out_filename,overwrite=True,include_optimizer=True)
        mlflow.log_artifact(out_filename)
    ### end if
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
