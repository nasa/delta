"""
Script test out the image chunk generation calls.
"""
#pylint: disable=C0413
import sys
import os
import argparse
import functools
# import matplotlib.pyplot as plt
# import numpy as np

### Tensorflow includes

import mlflow

import tensorflow as tf
# from tensorflow import keras
#import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from delta import config #pylint: disable=C0413
from delta.imagery import imagery_dataset #pylint: disable=C0413
from delta.ml.train import Experiment  #pylint: disable=C0413
from delta.ml.networks import make_task_specific



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
def assemble_dataset(config_values):

    # Use wrapper class to create a Tensorflow Dataset object.
    # - The dataset will provide image chunks and corresponding labels.
    ids = imagery_dataset.ImageryDatasetTFRecord(config_values)
    ds = ids.dataset()
    ds = ds.repeat(config_values['ml']['num_epochs']).batch(config_values['ml']['batch_size'])
    ds = ds.prefetch(None)

    return ds


def assemble_dataset_for_predict(config_values):
    # Slightly simpler version of the previous function
    ids = imagery_dataset.ImageryDatasetTFRecord(config_values)
    ds  = ids.dataset().batch(1) # Batch needed to match the original format
    return ds


def main(argsIn): #pylint: disable=R0914

    usage  = "usage: train_autoencoder [options]"
    parser = argparse.ArgumentParser(usage=usage)

    parser = argparse.ArgumentParser(usage='train_autoencoder.py [options]')

    parser.add_argument("--config-file", dest="config_file", default=None,
                        help="Dataset configuration file.")
    parser.add_argument("--data-folder", dest="data_folder", default=None,
                        help="Specify data folder instead of supplying config file.")
    parser.add_argument("--image-type", dest="image_type", default=None,
                        help="Specify image type along with the data folder."
                        +"(landsat, landsat-simple, worldview, or rgba)")

    parser.add_argument("--num-debug-images", dest="num_debug_images", default=30, type=int,
                        help="Run this many images through the AE after training and write the "
                        "input/output pairs to disk.")

    parser.add_argument("--num-gpus", dest="num_gpus", default=0, type=int,
                        help="Try to use this many GPUs.")

    try:
        options = parser.parse_args(argsIn)
    except argparse.ArgumentError:
        print(usage)
        return -1

    config.load_config_file(options.config_file)
    config_values = config.get_config()

    output_folder = config_values['ml']['output_folder']
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print('loading data from ' + config_values['input_dataset']['data_directory'])
    ids = imagery_dataset.ImageryDatasetTFRecord(config_values)

    # TF additions
    # If the mlfow directory doesn't exist, create it.
    mlflow_tracking_dir = os.path.join(output_folder,'mlruns')
    if not os.path.exists(mlflow_tracking_dir):
        os.mkdir(mlflow_tracking_dir)
    ### end if

    print('Creating experiment')
    experiment = Experiment(mlflow_tracking_dir,
                            'task_specific_%s'%(config_values['input_dataset']['image_type'],),
                            loss_fn='categorical_crossentropy',
                            output_dir=output_folder)
    mlflow.log_param('image type',   config_values['input_dataset']['image_type'])
    mlflow.log_param('image folder', config_values['input_dataset']['data_directory'])
    mlflow.log_param('chunk size',   config_values['ml']['chunk_size'])
    print('Creating model')
    in_data_shape = (ids.chunk_size(), ids.chunk_size(), ids.num_bands())
    out_data_shape = config_values['ml']['num_classes']
    print('Training')

    # Estimator interface requires the dataset to be constructed within a function.
#    tf.logging.set_verbosity(tf.logging.INFO)

    dataset_fn = functools.partial(assemble_dataset, config_values)
    model_fn = functools.partial(make_task_specific, in_data_shape, out_data_shape)

    model = experiment.train_keras(model_fn, dataset_fn,
                                   num_epochs=config_values['ml']['num_epochs'],
                                   num_gpus=options.num_gpus)

    print(model) # Need to do something with the estimator to appease the lint gods
    print('Saving Model')
    if config_values['ml']['model_dest_name'] is not None:
        out_filename = os.path.join(output_folder, 
                                    config_values['ml']['model_dest_name'])
        tf.keras.models.save_model(model, out_filename, overwrite=True, 
                                   include_optimizer=True)
        mlflow.log_artifact(out_filename)
    ### end if

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
