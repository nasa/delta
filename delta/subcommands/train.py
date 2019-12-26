import sys

import mlflow

import tensorflow as tf
from tensorflow import keras

from delta.config import config
from delta.imagery import imagery_dataset
from delta.ml.train import Experiment

def setup_parser(subparsers):
    sub = subparsers.add_parser('train', description='Train a task-specific classifier.')
    sub.add_argument('model', help='File to save the network to.')
    sub.set_defaults(function=main)
    config.setup_arg_parser(sub)

def main(options):
    images = config.images()
    labels = config.labels()
    if len(images) == 0:#pylint:disable=len-as-condition
        print('No images specified.', file=sys.stderr)
        return 1
    if len(labels) == 0:#pylint:disable=len-as-condition
        print('No labels specified.', file=sys.stderr)
        return 1
    ids = imagery_dataset.ImageryDataset(images, labels, config.chunk_size(), config.chunk_stride())

    experiment = Experiment('task_specific_%s'%(images.type()),
                            loss_fn=config.loss_function())
    mlflow.log_param('image type',   images.type())
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
                                      num_gpus=config.gpus())

    model.save(options.model)
    mlflow.log_artifact(options.model)

    return 0
