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


# With TF 1.12, the dataset needs to be constructed inside a function passed in to
# the estimator "train_and_evaluate" function to avoid getting a graph error!
def assemble_dataset():

    # Use wrapper class to create a Tensorflow Dataset object.
    # - The dataset will provide image chunks and corresponding labels.
    tc = config.training()
    images = config.images()
    ids = imagery_dataset.AutoencoderDataset(images, config.chunk_size(), tc.chunk_stride)
    ds = ids.dataset()
    ds = ds.repeat(config.epochs()).batch(config.batch_size())
    ds = ds.prefetch(None)

    return ds


def assemble_dataset_for_predict():
    # Slightly simpler version of the previous function
    tc = config.training()
    images = config.images()
    ids = imagery_dataset.AutoencoderDataset(images, config.chunk_size(), tc.chunk_stride)
    ds  = ids.dataset().batch(1) # Batch needed to match the original format
    return ds

def setup_parser(subparsers):
    sub = subparsers.add_parser('autoencode', help='Train an autoencder.')
    sub.add_argument("--num-debug-images", dest="num_debug_images", default=30, type=int,
                     help="Run this many images through the AE after training and write the "
                     "input/output pairs to disk.")
    sub.add_argument("--model", dest="model", required=True,
                     help="Location to save the model.")
    sub.set_defaults(function=main)
    config.setup_arg_parser(sub, labels=False, train=True)

def main(options):

    tc = config.training()
    images = config.images()
    aeds = imagery_dataset.AutoencoderDataset(images, config.chunk_size(), tc.chunk_stride)

    print('Creating experiment')
#    mlflow.log_param('image type',   config.image_type())
#    mlflow.log_param('image folder', config.data_directory())
#    mlflow.log_param('chunk size',   config.chunk_size())
    print('Creating model')
    data_shape = (aeds.chunk_size(), aeds.chunk_size(), aeds.num_bands())

    print('Training')
    # Estimator interface requires the dataset to be constructed within a function.
#     tf.logging.set_verbosity(tf.logging.INFO) # TODO 2.0

    # To do distribution of training with TF2/Keras, we need to create the model
    # in the scope of the distribution strategy (occurrs in the training function)
    model_fn = functools.partial(make_autoencoder,
                                 data_shape,
                                 encoding_size=300,
                                 encoder_type='conv',
                                 )

    model, _ = train(model_fn, aeds, tc)


    print('Saving Model')
    out_filename = options.model
    #out_filename = os.path.join(config.output_folder(), options.model)
    tf.keras.models.save_model(model, options.model, overwrite=True, include_optimizer = True)
    mlflow.log_artifact(out_filename)
    return 0
