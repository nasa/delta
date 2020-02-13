"""
Autoencoder.
"""

import functools

import mlflow
import tensorflow as tf

from delta.config import config
from delta.imagery import imagery_dataset
from delta.ml.train import train
from delta.ml.networks import make_autoencoder

def setup_parser(subparsers):
    sub = subparsers.add_parser('autoencode', help='Train an autoencder.')
    sub.add_argument("--model", dest="model", required=True,
                     help="Location to save the model.")
    sub.set_defaults(function=main)
    config.setup_arg_parser(sub, labels=False, train=True)

def main(options):

    tc = config.training()
    images = config.images()
    aeds = imagery_dataset.AutoencoderDataset(images, config.chunk_size(), tc.chunk_stride)

    data_shape = (aeds.chunk_size(), aeds.chunk_size(), aeds.num_bands())

    model_fn = functools.partial(make_autoencoder,
                                 data_shape,
                                 encoding_size=300,
                                 encoder_type='conv',
                                 )

    model, _ = train(model_fn, aeds, tc)

    out_filename = options.model
    tf.keras.models.save_model(model, options.model, overwrite=True, include_optimizer = True)
    mlflow.log_artifact(out_filename)
    return 0
