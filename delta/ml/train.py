# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Train neural networks.
"""

import datetime
import os
import tempfile
import shutil

import mlflow
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

from delta.config import config
from delta.imagery.imagery_dataset import ImageryDataset
from delta.imagery.imagery_dataset import AutoencoderDataset
from .io import save_model, load_model, print_network
from .config_parser import config_callbacks, loss_from_dict, metric_from_dict, \
                           optimizer_from_dict, config_augmentation

class DeltaLayer(Layer):
    """
    Network layer class with extra features specific to DELTA.

    Extentds `tensorflow.keras.layers.Layer`.
    """
    def callback(self): # pylint:disable=no-self-use
        """
        Override this method to make a layer automatically register
        a training callback.

        Returns
        -------
        tensorflow.keras.callbacks.Callback:
            The callback to register (or None).
        """
        return None

def _devices(num_gpus):
    '''
    Takes a number of GPUs and returns a list of TensorFlow LogicalDevices.

    Arguments

    num_gpus -- Number of GPUs to use.  If negative, will use all CPUs available.
    '''
    devs = None
    if num_gpus == 0:
        devs = [x.name for x in tf.config.list_logical_devices('CPU')]
    else:
        devs = [x.name for x in tf.config.list_logical_devices('GPU')]
        assert len(devs) >= num_gpus,\
               "Requested %d GPUs with only %d available." % (num_gpus, len(devs))
        if num_gpus > 0:
            devs = devs[:num_gpus]
    return devs

def _strategy(devices):
    '''Given a list of TensorFlow Logical Devices, returns a distribution strategy.'''
    strategy = None
    if len(devices) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device=devices[0])
    else:
        strategy = tf.distribute.MirroredStrategy(devices=devices)
    return strategy

def _prep_datasets(ids, tc):
    ds = ids.dataset(config.dataset.classes.weights(), config_augmentation())

    validation=None
    if tc.validation:
        if tc.validation.from_training:
            validation = ds.take(tc.validation.steps)
            ds = ds.skip(tc.validation.steps)
        else:
            vimg   = tc.validation.images
            vlabel = tc.validation.labels
            if not vimg:
                validation = None
            else:
                if vlabel:
                    vimagery = ImageryDataset(vimg, vlabel, ids.output_shape(), ids.chunk_shape(),
                                              tile_shape=ids.tile_shape(), stride=ids.stride(),
                                              tile_overlap=ids.tile_overlap())
                else:
                    vimagery = AutoencoderDataset(vimg, ids.chunk_shape(), tile_shape=ids.tile_shape(),
                                                  stride=ids.stride(), tile_overlap=ids.tile_overlap())
                validation = vimagery.dataset(config.dataset.classes.weights())
                if tc.validation.steps:
                    validation = validation.take(tc.validation.steps)
        if validation:
            validation = validation.batch(tc.batch_size, drop_remainder=True).prefetch(1)
    else:
        validation = None

    ds = ds.batch(tc.batch_size, drop_remainder=True)
    ds = ds.prefetch(1)
    if tc.steps:
        ds = ds.take(tc.steps)
    return (ds, validation)

def _log_mlflow_params(model, dataset, training_spec):
    images = dataset.image_set()
    #labels = dataset.label_set()
    mlflow.log_param('Images - Type',   images.type())
    mlflow.log_param('Images - Count',   len(images))
    mlflow.log_param('Images - Stride', training_spec.stride)
    mlflow.log_param('Images - Tile Size', len(model.layers))
    mlflow.log_param('Train - Steps', training_spec.steps)
    mlflow.log_param('Train - Loss Function', training_spec.loss)
    mlflow.log_param('Train - Epochs', training_spec.epochs)
    mlflow.log_param('Train - Batch Size', training_spec.batch_size)
    mlflow.log_param('Train - Optimizer', training_spec.optimizer)
    mlflow.log_param('Model - Layers', len(model.layers))
    mlflow.log_param('Model - Parameters - Non-Trainable',
                     np.sum([K.count_params(w) for w in model.non_trainable_weights]))
    mlflow.log_param('Model - Parameters - Trainable',
                     np.sum([K.count_params(w) for w in model.trainable_weights]))
    mlflow.log_param('Model - Shape - Output',   dataset.output_shape())
    mlflow.log_param('Model - Shape - Input',   dataset.input_shape())
    #mlflow.log_param('Status', 'Running') Illegal to change the value!

class _EpochResetCallback(tf.keras.callbacks.Callback):
    """
    Reset imagery_dataset file counts on epoch end
    """
    def __init__(self, ids, stop_epoch):
        super().__init__()
        self.ids = ids
        self.last_epoch = stop_epoch - 1

    def on_epoch_end(self, epoch, _=None):
        if config.general.verbose():
            print('Finished epoch ' + str(epoch))
        # Leave the counts from the last epoch just as a record
        if epoch != self.last_epoch:
            self.ids.reset_access_counts()

class _MLFlowCallback(tf.keras.callbacks.Callback):
    """
    Callback to log everything for MLFlow.
    """
    def __init__(self, temp_dir):
        super().__init__()
        self.epoch = 0
        self.batch = 0
        self.temp_dir = temp_dir

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        for k in logs.keys():
            if k.startswith('val_'):
                mlflow.log_metric('Validation ' + k[4:], logs[k], epoch)
            else:
                mlflow.log_metric('Epoch ' + k, logs[k], epoch)

    def on_train_batch_end(self, batch, logs=None):
        self.batch = batch
        if batch % config.mlflow.frequency() == 0:
            for k in logs.keys():
                if k in ('batch', 'size'):
                    continue
                mlflow.log_metric(k, logs[k], step=batch)
        if config.mlflow.checkpoints.frequency() and batch % config.mlflow.checkpoints.frequency() == 0:
            config.train.default_model_extension()
            filename = os.path.join(self.temp_dir, '%d%s' % (batch, config.train.default_model_extension()))
            save_model(self.model, filename)
            if config.mlflow.checkpoints.only_save_latest():
                old = filename
                filename = os.path.join(self.temp_dir, 'latest' + config.train.default_model_extension())
                os.rename(old, filename)
            mlflow.log_artifact(filename, 'checkpoints')
            if os.path.isdir(filename):
                shutil.rmtree(filename)
            else:
                os.remove(filename)

def _mlflow_train_setup(model, dataset, training_spec):
    mlflow.set_tracking_uri(config.mlflow.uri())
    mlflow.set_experiment(config.mlflow.experiment())
    mlflow.start_run()
    _log_mlflow_params(model, dataset, training_spec)

    temp_dir = tempfile.mkdtemp()
    fname = os.path.join(temp_dir, 'config.yaml')
    with open(fname, 'w') as f:
        f.write(config.export())
    mlflow.log_artifact(fname)
    os.remove(fname)

    return _MLFlowCallback(temp_dir)

def _build_callbacks(model, dataset, training_spec):
    """
    Create callbacks needed based on configuration.

    Returns (list of callbacks, mlflow callback).
    """
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    # add callbacks from DeltaLayers
    for l in model.layers:
        if isinstance(l, DeltaLayer):
            c = l.callback()
            if c:
                callbacks.append(c)

    mcb = None
    if config.mlflow.enabled():
        mcb = _mlflow_train_setup(model, dataset, training_spec)
        callbacks.append(mcb)
        if config.general.verbose():
            print('Using mlflow folder: ' + mlflow.get_artifact_uri())

    if config.tensorboard.enabled():
        tb_dir = config.tensorboard.dir()
        if config.mlflow.enabled():
            tb_dir = os.path.join(tb_dir, str(mlflow.active_run().info.run_id))
            mlflow.log_param('TensorBoard Directory', tb_dir)
        else:
            tb_dir = os.path.join(tb_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tcb = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                             update_freq='epoch',
                                             histogram_freq=1,
                                             write_images=True,
                                             embeddings_freq=1)
        callbacks.append(tcb)

    callbacks.append(_EpochResetCallback(dataset, training_spec.epochs))

    callbacks.extend(config_callbacks())

    return (callbacks, mcb)

def _compile_helper(model, training_spec):
    model.compile(optimizer=optimizer_from_dict(training_spec.optimizer),
                  loss=loss_from_dict(training_spec.loss),
                  metrics=[metric_from_dict(m) for m in training_spec.metrics])

class ContinueTrainingException(Exception):
    """
    Callbacks can raise this exception to modify the model, recompile, and
    continue training.
    """
    def __init__(self, msg: str=None, completed_epochs: int=0,
                 recompile_model: bool=False, learning_rate: float=None):
        """
        Parameters
        ----------
        msg: str
            Optional error message.
        completed_epochs: int
            The number of epochs that have been finished. (resumes from the next epoch)
        recompile_model: bool
            If True, recompile the model. This is necessary if the model has been changed.
        learning_rate: float
            Optionally set the learning rate to the given value.
        """
        super().__init__(msg)
        self.completed_epochs = completed_epochs
        self.recompile_model = recompile_model
        self.learning_rate = learning_rate

def compile_model(model_fn, training_spec, resume_path=None):
    """
    Compile and check that the model is valid.

    Parameters
    ----------
    model_fn: Callable[[], tensorflow.keras.model.Model]
        Function to construct a keras Model.
    training_spec: delta.ml.ml_config.TrainingSpec
        Trainnig parameters.
    resume_path: str
        File name to load initial model weights from.

    Returns
    -------
    tensorflow.keras.models.Model:
        The compiled model, ready for training.
    """
    if not hasattr(training_spec, 'strategy'):
        training_spec.strategy = _strategy(_devices(config.general.gpus()))
    with training_spec.strategy.scope():
        model = model_fn()
        assert isinstance(model, tf.keras.models.Model), \
                "Model is not a Tensorflow Keras model"

        if resume_path is not None:
            print('Loading existing model: ' + resume_path)
            if resume_path.endswith('.h5'):
                model.load_weights(resume_path)
            else: # SavedModel format
                model = load_model(resume_path)

        _compile_helper(model, training_spec)

    input_shape = model.input_shape
    output_shape = model.output_shape

    assert len(input_shape) == 4, 'Input to network is wrong shape.'
    assert input_shape[0] is None, 'Input is not batched.'
    # The below may no longer be valid if we move to convolutional architectures.
    assert input_shape[1] == input_shape[2], 'Input to network is not chunked'
    assert len(output_shape) == 2 or output_shape[1] == output_shape[2], 'Output from network is not chunked'

    if config.general.verbose():
        print('Training model:')
        print_network(model, (512, 512, 8))
        print(model.summary(line_length=120))

    return model

def train(model_fn, dataset : ImageryDataset, training_spec, resume_path=None):
    """
    Trains the specified model on a dataset according to a training
    specification.

    Parameters
    ----------
    model_fn: Callable[[], tensorflow.keras.model.Model]
        Function that constructs a model.
    dataset: delta.imagery.imagery_dataset.ImageryDataset
        Dataset to train on.
    training_spec: delta.ml.ml_config.TrainingSpec
        Training parameters.
    resume_path: str
        Optional file to load initial model weights from.

    Returns
    -------
    (tensorflow.keras.models.Model, History):
        The trained model and the training history.
    """
    model = compile_model(model_fn, training_spec, resume_path)
    assert model.input_shape[3] == dataset.num_bands(), 'Number of bands in model does not match data.'
    # last element differs for the sparse metrics
    assert model.output_shape[1:-1] == dataset.output_shape()[:-1] or (model.output_shape[1] is None), \
            'Network output shape %s does not match label shape %s.' % \
            (model.output_shape[1:], dataset.output_shape()[:-1])

    (ds, validation) = _prep_datasets(dataset, training_spec)

    (callbacks, mcb) = _build_callbacks(model, dataset, training_spec)

    try:

        # Mark that we need to check the dataset counts the
        # first time we try to read the images.
        # This won't do anything unless we are resuming training.
        dataset.reset_access_counts(set_need_check=True)

        if (training_spec.steps is None) or (training_spec.steps > 0):
            done = False
            epochs = training_spec.epochs
            initial_epoch = 0
            while not done:
                try:
                    history = model.fit(ds,
                                        epochs=epochs,
                                        initial_epoch=initial_epoch,
                                        callbacks=callbacks,
                                        validation_data=validation,
                                        validation_steps=None, # Steps are controlled in the dataset setup
                                        steps_per_epoch=None,
                                        verbose=1) # Set to 2 when logging
                    done = True
                except ContinueTrainingException as cte:
                    print('Recompiling model and resuming training.')
                    initial_epoch += cte.completed_epochs
                    if cte.recompile_model:
                        model = compile_model(model, training_spec)
                    if cte.learning_rate:
                        K.set_value(model.optimizer.lr, cte.learning_rate)
        else: # Skip training
            print('Skipping straight to validation')
            history = model.evaluate(validation, steps=training_spec.validation.steps,
                                     callbacks=callbacks, verbose=1)

        if config.mlflow.enabled():
            model_path = os.path.join(mcb.temp_dir, 'final_model' + config.train.default_model_extension())
            print('\nFinished, saving model to %s.' % (mlflow.get_artifact_uri() + '/final_model' + config.train.default_model_extension()))
            save_model(model, model_path)
            mlflow.log_artifact(model_path)
            if os.path.isdir(model_path):
                shutil.rmtree(model_path)
            else:
                os.remove(model_path)
            mlflow.log_param('Status', 'Completed')
    except:
        if config.mlflow.enabled():
            mlflow.log_param('Status', 'Aborted')
            mlflow.log_param('Epoch', mcb.epoch)
            mlflow.log_param('Batch', mcb.batch)
            mlflow.end_run('FAILED')
            model_path = os.path.join(mcb.temp_dir, 'aborted_model' + config.train.default_model_extension())
            print('\nAborting, saving current model to %s.' % (mlflow.get_artifact_uri() + '/aborted_model' + config.train.default_model_extension()))
            save_model(model, model_path)
            mlflow.log_artifact(model_path)
            if os.path.isdir(model_path):
                shutil.rmtree(model_path)
            else:
                os.remove(model_path)
        raise
    finally:
        if config.mlflow.enabled():
            if mcb and mcb.temp_dir:
                shutil.rmtree(mcb.temp_dir)

    if config.mlflow.enabled():
        mlflow.end_run()

    return model, history
