"""
Script test out the image chunk generation calls.
"""
import sys
import os
import argparse

import mlflow

import tensorflow as tf
from tensorflow import keras

from delta.config import config
from delta.imagery import imagery_dataset
from delta.ml.train import Experiment

def main(argsIn):
    parser = argparse.ArgumentParser(usage='train_task_specific.py [options]')

    parser.add_argument('network_file', help='File to save the network to.')
    options = config.parse_args(parser, argsIn)

    if config.output_folder() and not os.path.exists(config.output_folder()):
        os.mkdir(config.output_folder())

    config_d = config.dataset()
    ids = imagery_dataset.ImageryDataset(config_d, config.chunk_size(), config.chunk_stride())

    # TF additions
    # If the mlfow directory doesn't exist, create it.
    mlflow_tracking_dir = os.path.join(config.output_folder(),'mlruns')
    if not os.path.exists(mlflow_tracking_dir):
        os.mkdir(mlflow_tracking_dir)
    ### end if

    experiment = Experiment(mlflow_tracking_dir,
                            'task_specific_%s'%(config_d.image_type()),
                            loss_fn=config.loss_function(),
                            output_dir=config.output_folder())
    mlflow.log_param('image type',   config_d.image_type())
    mlflow.log_param('image folder', config_d.data_directory())
    mlflow.log_param('chunk size',   config.chunk_size())
    in_data_shape = (ids.chunk_size(), ids.chunk_size(), ids.num_bands())
    out_data_shape = config.num_classes()

    def make_dumb_network():
        return keras.Sequential([
            keras.layers.Flatten(input_shape=in_data_shape),
            keras.layers.Dense(in_data_shape[0] * in_data_shape[1] * in_data_shape[2], activation=tf.nn.relu),
            keras.layers.Dense(out_data_shape, activation=tf.nn.softmax)
            ])

    ds = ids.dataset()
    ds = ds.batch(config.batch_size()).repeat(config.num_epochs()).take(50000)
    model, _ = experiment.train_keras(make_dumb_network, ds,
                                      num_gpus=config.num_gpus())

    model.save(options.network_file)
    mlflow.log_artifact(options.network_file)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
