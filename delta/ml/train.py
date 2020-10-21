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

import os
import tempfile
import shutil

import mlflow
import tensorflow as tf
from tensorflow.keras.layers import Layer

from delta.config import config
from delta.imagery.imagery_dataset import ImageryDataset
from delta.imagery.imagery_dataset import AutoencoderDataset
from .io import save_model
from .config_parser import config_callbacks, loss_from_dict, metric_from_dict, optimizer_from_dict

class DeltaLayer(Layer):
    """
    Network layer class with extra features specific to DELTA.

    Extentds `tensorflow.keras.layers.Layer`.
    """
    def callback(self): # pylint:disable=no-self-use
        """
        Returns a Keras callback to be added, or None.
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

def _prep_datasets(ids, tc, chunk_size, output_size):
    ds = ids.dataset(config.dataset.classes.weights())

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
                    vimagery = ImageryDataset(vimg, vlabel, chunk_size, output_size, tc.chunk_stride,
                                              resume_mode=False)
                else:
                    vimagery = AutoencoderDataset(vimg, chunk_size, tc.chunk_stride, resume_mode=False)
                validation = vimagery.dataset(config.dataset.classes.weights())
                if tc.validation.steps:
                    validation = validation.take(tc.validation.steps)
        if validation:
            validation = validation.batch(tc.batch_size)
    else:
        validation = None

    ds = ds.batch(tc.batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    if tc.steps:
        ds = ds.take(tc.steps)
    ds = ds.repeat(tc.epochs)
    return (ds, validation)

def _log_mlflow_params(model, dataset, training_spec):
    images = dataset.image_set()
    #labels = dataset.label_set()
    mlflow.log_param('Image Type',   images.type())
    mlflow.log_param('Preprocess',   images.preprocess())
    mlflow.log_param('Number of Images',   len(images))
    mlflow.log_param('Chunk Size',   model.input_shape[1])
    mlflow.log_param('Chunk Stride', training_spec.chunk_stride)
    mlflow.log_param('Output Shape',   dataset.output_shape())
    mlflow.log_param('Steps', training_spec.steps)
    mlflow.log_param('Loss Function', training_spec.loss)
    mlflow.log_param('Epochs', training_spec.epochs)
    mlflow.log_param('Batch Size', training_spec.batch_size)
    mlflow.log_param('Optimizer', training_spec.optimizer)
    mlflow.log_param('Model Layers', len(model.layers))
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

    def on_epoch_end(self, epoch, _=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        self.batch = batch
        if batch % config.mlflow.frequency() == 0:
            for k in logs.keys():
                if k in ('batch', 'size'):
                    continue
                mlflow.log_metric(k, logs[k], step=batch)
        if config.mlflow.checkpoints.frequency() and batch % config.mlflow.checkpoints.frequency() == 0:
            filename = os.path.join(self.temp_dir, '%d.h5' % (batch))
            save_model(self.model, filename)
            if config.mlflow.checkpoints.only_save_latest():
                old = filename
                filename = os.path.join(self.temp_dir, 'latest.h5')
                os.rename(old, filename)
            mlflow.log_artifact(filename, 'checkpoints')
            os.remove(filename)

    def on_test_batch_end(self, _, logs=None): # pylint:disable=no-self-use
        for k in logs.keys():
            if k in ('batch', 'size'):
                continue
            mlflow.log_metric('validation_' + k, logs[k])

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
    if config.tensorboard.enabled():
        tcb = tf.keras.callbacks.TensorBoard(log_dir=config.tensorboard.dir(),
                                             update_freq='epoch',
                                             histogram_freq=1,
                                             write_images=True,
                                             embeddings_freq=1)
        callbacks.append(tcb)

    mcb = None
    if config.mlflow.enabled():
        mcb = _mlflow_train_setup(model, dataset, training_spec)
        callbacks.append(mcb)
        if config.general.verbose():
            print('Using mlflow folder: ' + mlflow.get_artifact_uri())

    callbacks.append(_EpochResetCallback(dataset, training_spec.epochs))

    callbacks.extend(config_callbacks())

    return (callbacks, mcb)

def _compile_model(model_fn, dataset, training_spec):
    """
    Compile and check that the model is valid.
    """
    # This does not work when using the CPU!
    if isinstance(model_fn, tf.keras.Model):
        model = model_fn
        print('WARNING: Model loaded without TF strategy/device wrapper, this may fail in some configurations!')
        # May not be able to improve on this until tf 2.2
    else:
        with _strategy(_devices(config.general.gpus())).scope():
            model = model_fn()
            assert isinstance(model, tf.keras.models.Model),\
                   "Model is not a Tensorflow Keras model"
            model.compile(optimizer=optimizer_from_dict(training_spec.optimizer),
                          loss=loss_from_dict(training_spec.loss),
                          metrics=[metric_from_dict(m) for m in training_spec.metrics])

    input_shape = model.input_shape
    output_shape = model.output_shape

    assert len(input_shape) == 4, 'Input to network is wrong shape.'
    assert input_shape[0] is None, 'Input is not batched.'
    # The below may no longer be valid if we move to convolutional architectures.
    assert input_shape[1] == input_shape[2], 'Input to network is not chunked'
    assert len(output_shape) == 2 or output_shape[1] == output_shape[2], 'Output from network is not chunked'
    assert input_shape[3] == dataset.num_bands(), 'Number of bands in model does not match data.'
    # last element differs for the sparse metrics
    assert output_shape[1:-1] == dataset.output_shape()[:-1] or (output_shape[1] is None), \
            'Network output shape %s does not match label shape %s.' % (output_shape[1:], dataset.output_shape()[:-1])

    if config.general.verbose():
        print('Training model:')
        print(model.summary())

    return model

def train(model_fn, dataset : ImageryDataset, training_spec):
    """
    Trains the specified model on a dataset according to a training
    specification.
    """
    model = _compile_model(model_fn, dataset, training_spec)

    (ds, validation) = _prep_datasets(dataset, training_spec, model.input_shape[1], model.output_shape[1])

    (callbacks, mcb) = _build_callbacks(model, dataset, training_spec)

    try:

        # Mark that we need to check the dataset counts the
        # first time we try to read the images.
        # This won't do anything unless we are resuming training.
        dataset.reset_access_counts(set_need_check=True)

        if (training_spec.steps is None) or (training_spec.steps > 0):
            history = model.fit(ds,
                                epochs=training_spec.epochs,
                                callbacks=callbacks,
                                validation_data=validation,
                                validation_steps=training_spec.validation.steps if training_spec.validation else None,
                                steps_per_epoch=training_spec.steps,
                                verbose=1)
        else: # Skip training
            print('Skipping straight to validation')
            history = model.evaluate(validation, steps=training_spec.validation.steps,
                                     callbacks=callbacks, verbose=1)

        if config.mlflow.enabled():
            model_path = os.path.join(mcb.temp_dir, 'final_model.h5')
            print('\nFinished, saving model to %s.' % (mlflow.get_artifact_uri() + '/final_model.h5'))
            save_model(model, model_path)
            mlflow.log_artifact(model_path)
            os.remove(model_path)
            mlflow.log_param('Status', 'Completed')
    except:
        if config.mlflow.enabled():
            mlflow.log_param('Status', 'Aborted')
            mlflow.end_run('FAILED')
            model_path = os.path.join(mcb.temp_dir, 'aborted_model.h5')
            print('\nAborting, saving current model to %s.' % (mlflow.get_artifact_uri() + '/aborted_model.h5'))
            save_model(model, model_path)
            mlflow.log_artifact(model_path)
            os.remove(model_path)
        raise
    finally:
        if config.mlflow.enabled():
            mlflow.log_param('Epoch', mcb.epoch)
            mlflow.log_param('Batch', mcb.batch)
            if mcb and mcb.temp_dir:
                shutil.rmtree(mcb.temp_dir)

    if config.mlflow.enabled():
        mlflow.end_run()

    return model, history
