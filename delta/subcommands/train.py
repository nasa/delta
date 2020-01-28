import sys

import tensorflow as tf
from tensorflow import keras

from delta.config import config
from delta.imagery.imagery_dataset import ImageryDataset
from delta.ml.train import train

def setup_parser(subparsers):
    sub = subparsers.add_parser('train', help='Train a task-specific classifier.')
    sub.add_argument('model', help='File to save the network to.')
    sub.set_defaults(function=main)
    config.setup_arg_parser(sub, train=True)

def main(options):
    images = config.images()
    labels = config.labels()
    if not images:
        print('No images specified.', file=sys.stderr)
        return 1
    if not labels:
        print('No labels specified.', file=sys.stderr)
        return 1
    tc = config.training()

    ids = ImageryDataset(images, labels, config.chunk_size(), tc.chunk_stride)

    in_data_shape = (config.chunk_size(), config.chunk_size(), ids.num_bands())
    out_data_shape = config.classes()

    def make_dumb_network():
#        return keras.Sequential([
#            keras.layers.Flatten(input_shape=in_data_shape),
#            keras.layers.Dense(in_data_shape[0] * in_data_shape[1] * in_data_shape[2], activation=tf.nn.relu),
#            keras.layers.Dense(out_data_shape, activation=tf.nn.softmax)
#            ])
        return keras.Sequential([
            keras.layers.InputLayer(input_shape=in_data_shape),
            keras.layers.Conv2D(100, (5, 5), activation='relu', padding='same'),
            keras.layers.Dropout(0.3),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(100, (3, 3), activation='relu', padding='same'),
            keras.layers.Dropout(0.3),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(out_data_shape, activation=tf.nn.softmax)
            ])

    model, _ = train(make_dumb_network, ids, tc)

    model.save(options.model)

    return 0
