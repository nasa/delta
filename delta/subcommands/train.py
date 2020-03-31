"""
Train a neural network.
"""

import sys

import tensorflow as tf

from delta.config import config
from delta.imagery import imagery_dataset
from delta.ml.train import train
from delta.ml.model_parser import config_model
from delta.ml.layers import ALL_LAYERS

def setup_parser(subparsers):
    sub = subparsers.add_parser('train', help='Train a task-specific classifier.')
    sub.add_argument('--autoencoder', action='store_true',
                     help='Train autoencoder (ignores labels).')
    sub.add_argument('--resume', help='Use the model as a starting point for the training.')
    sub.add_argument('model', nargs='?', default=None, help='File to save the network to.')
    sub.set_defaults(function=main)
    config.setup_arg_parser(sub, train=True)

def main(options):
    images = config.images()
    if not images:
        print('No images specified.', file=sys.stderr)
        return 1
    tc = config.training()
    if options.autoencoder:
        ids = imagery_dataset.AutoencoderDataset(images, config.chunk_size(), tc.chunk_stride)
    else:
        labels = config.labels()
        if not labels:
            print('No labels specified.', file=sys.stderr)
            return 1
        ids = imagery_dataset.ImageryDataset(images, labels, config.chunk_size(),
                                             config.output_size(), tc.chunk_stride)

    try:
        if options.resume is not None:
            model = tf.keras.models.load_model(options.resume, custom_objects=ALL_LAYERS)
        else:
            model = config_model(ids.num_bands())
        model, _ = train(model, ids, tc)

        if options.model is not None:
            model.save(options.model)
    except KeyboardInterrupt:
        print()
        print('Training cancelled.')

    return 0
