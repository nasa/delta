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

from delta.config import config #pylint: disable=C0413
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
def assemble_dataset():
    # Use wrapper class to create a Tensorflow Dataset object.
    # - The dataset will provide image chunks and corresponding labels.
    ids = imagery_dataset.AutoencoderDataset(config.dataset(), config.chunk_size(), config.chunk_stride())
    ds = ids.dataset()
    ds = ds.repeat(config.num_epochs()).batch(config.batch_size())
    ds = ds.prefetch(None)

    return ds

def main(argsIn): #pylint: disable=R0914
    parser = argparse.ArgumentParser(usage='train_autoencoder.py [options]')

    config.parse_args(parser, argsIn)

    if not os.path.exists(config.output_folder()):
        os.mkdir(config.output_folder())

    config_d = config.dataset()
    ids = imagery_dataset.ImageryDataset(config_d, config.chunk_size(), config.chunk_stride())

    # TF additions
    # If the mlfow directory doesn't exist, create it.
    mlflow_tracking_dir = os.path.join(config.output_folder(),'mlruns')
    if not os.path.exists(mlflow_tracking_dir):
        os.mkdir(mlflow_tracking_dir)
    ### end if

    print('Creating experiment')
    experiment = Experiment(mlflow_tracking_dir,
                            'task_specific_%s'%(config_d.image_type()),
                            loss_fn='categorical_crossentropy',
                            output_dir=config.output_folder())
    mlflow.log_param('image type',   config_d.image_type())
    mlflow.log_param('image folder', config_d.data_directory())
    mlflow.log_param('chunk size',   config.chunk_size())
    print('Creating model')
    in_data_shape = (ids.chunk_size(), ids.chunk_size(), ids.num_bands())
    out_data_shape = config.num_classes()
    print('Training')

    # Estimator interface requires the dataset to be constructed within a function.
#    tf.logging.set_verbosity(tf.logging.INFO)

    model_fn = functools.partial(make_task_specific, in_data_shape, out_data_shape)

    model = experiment.train_keras(model_fn, assemble_dataset,
                                   num_epochs=config.num_epochs(),
                                   num_gpus=config.num_gpus())

    print(model) # Need to do something with the estimator to appease the lint gods
    print('Saving Model')
    if config.model_dest_name() is not None:
        out_filename = os.path.join(config.output_folder(), config.model_dest_name())
        tf.keras.models.save_model(model, out_filename, overwrite=True,
                                   include_optimizer=True)
        mlflow.log_artifact(out_filename)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
