import sys

import functools

#import tensorflow as tf
#from tensorflow import keras

from delta.config import config
from delta.imagery.imagery_dataset import ImageryDataset
from delta.ml.train import train
from delta.ml.model_parser import model_from_yaml


def setup_parser(subparsers):
    sub = subparsers.add_parser('train', help='Train a task-specific classifier.')
    sub.add_argument('model', help='File to save the network to.')
    sub.set_defaults(function=main)
    config.setup_arg_parser(sub, train=True)

def main(options):
    images = config.images()
    labels = config.labels()
    model_desc_string = open(config.model_description(), 'r')
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

    #def make_dumb_network():
    #    return keras.Sequential([
    #        keras.layers.Flatten(input_shape=in_data_shape),
    #        keras.layers.Dense(in_data_shape[0] * in_data_shape[1] * in_data_shape[2], activation=tf.nn.relu),
    #        keras.layers.Dense(out_data_shape, activation=tf.nn.softmax)
    #        ])

    params_exposed = { 'out_shape' : out_data_shape, 'in_shape' : in_data_shape}

    model_fn = functools.partial(model_from_yaml,
                                 model_desc_string,
                                 params_exposed
                                 )


    #model = model_from_yaml(model_desc_string, params_exposed)

    #model, _ = train(make_dumb_network, ids, tc)
    model, _ = train(model_fn, ids, tc)

    model.save(options.model)

    return 0
