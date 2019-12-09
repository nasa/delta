"""
Script test out the image chunk generation calls.
"""
import sys
import os
import argparse
import functools

import mlflow

import tensorflow as tf

from delta.config import config
from delta.imagery import imagery_dataset
from delta.ml.train import Experiment
from delta.ml.networks import make_task_specific

def main(argsIn):
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
                            loss_fn=config.loss_function(),
                            output_dir=config.output_folder())
    mlflow.log_param('image type',   config_d.image_type())
    mlflow.log_param('image folder', config_d.data_directory())
    mlflow.log_param('chunk size',   config.chunk_size())
    print('Creating model')
    in_data_shape = (ids.chunk_size(), ids.chunk_size(), ids.num_bands())
    out_data_shape = config.num_classes()
    print(out_data_shape)
    print('Training')

    # Estimator interface requires the dataset to be constructed within a function.
#    tf.logging.set_verbosity(tf.logging.INFO)

    model_fn = functools.partial(make_task_specific, in_data_shape, out_data_shape)

    ds = ids.dataset(filter_zero=False, shuffle=False)
    ds = ds.batch(config.batch_size()).repeat(config.num_epochs())
    model, _ = experiment.train_keras(model_fn, ds,
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
    sys.exit(main(sys.argv))
