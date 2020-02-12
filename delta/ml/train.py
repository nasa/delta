import os
import tempfile
import shutil

import mlflow
import tensorflow as tf

from delta.config import config
from delta.imagery.imagery_dataset import ImageryDataset

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
    ds = ids.dataset()
    ds = ds.batch(tc.batch_size)
    if tc.validation:
        if tc.validation.from_training:
            validation = ds.take(tc.validation.steps)
            ds = ds.skip(tc.validation.steps)
        else:
            vimg = tc.validation.images
            vlabel = tc.validation.labels
            if not vimg or not vlabel:
                validation = None
            else:
                vimagery = ImageryDataset(vimg, vlabel, chunk_size, output_size, tc.chunk_stride)
                validation = vimagery.dataset().batch(tc.batch_size).take(tc.validation.steps)
    else:
        validation = None
    ds = ds.repeat(tc.epochs)
    if tc.steps:
        ds = ds.take(tc.steps)
    return (ds, validation)

def _log_mlflow_params(model, dataset, training_spec):
    images = dataset.image_set()
    #labels = dataset.label_set()
    mlflow.log_param('Image Type',   images.type())
    mlflow.log_param('Preprocess',   images.preprocess())
    mlflow.log_param('Number of Images',   len(images))
    mlflow.log_param('Chunk Size',   dataset.chunk_size())
    mlflow.log_param('Chunk Stride', training_spec.chunk_stride)
    mlflow.log_param('Output Size',   dataset.output_size())
    mlflow.log_param('Steps', training_spec.steps)
    mlflow.log_param('Loss Function', training_spec.loss_function)
    mlflow.log_param('Epochs', training_spec.epochs)
    mlflow.log_param('Batch Size', training_spec.batch_size)
    mlflow.log_param('Optimizer', training_spec.optimizer)
    mlflow.log_param('Model Layers', len(model.layers))

def train(model_fn, dataset, training_spec):
    '''
    Trains the specified model given the images, corresponding labels, and training specification.
    '''
    # TODO: Check that this checks the training spec for desired devices to run on.
    with _strategy(_devices(config.gpus())).scope():
        model = model_fn()
        assert isinstance(model, tf.keras.models.Model),\
               "Model is not a Tensorflow Keras model"
        # TODO: specify learning rate and optimizer parameters, change learning rate over time
        model.compile(optimizer=training_spec.optimizer, loss=training_spec.loss_function,
                      metrics=training_spec.metrics)

    input_shape = model.layers[0].input_shape
    output_shape = model.layers[-1].output_shape
    chunk_size = input_shape[1]

    assert len(input_shape) == 4, 'Input to network is wrong shape.' # TODO: Hard coding 4 a bad idea?
    assert input_shape[0] is None, 'Input is not batched.'
    # The below may no longer be valid if we move to convolutional architectures.
    assert input_shape[1] == input_shape[2], 'Input to network is not chunked'
    assert len(output_shape) == 2 or output_shape[1] == output_shape[2], 'Output from network is not chunked'
    assert input_shape[3] == dataset.num_bands(), 'Number of bands in model does not match data.'

    (ds, validation) = _prep_datasets(dataset, training_spec, chunk_size, output_shape[1])

    temp_dir = None
    callbacks = []
    if config.tb_enabled():
        cb = tf.keras.callbacks.TensorBoard(log_dir=config.tb_dir(),
                                            update_freq=config.tb_freq(),
                                           )
        callbacks.append(cb)

    if config.mlflow_enabled():
        mlflow.set_tracking_uri(config.mlflow_uri())
        mlflow.set_experiment(training_spec.experiment)
        mlflow.start_run()
        _log_mlflow_params(model, dataset, training_spec)

        def log_metrics_batch(batch, logs):
            if batch % config.mlflow_freq() != 0:
                return
            for k in logs.keys():
                if k in ('batch', 'size'):
                    continue
                mlflow.log_metric(k, logs[k], step=batch)
        def log_metrics_validate(_, logs):
            for k in logs.keys():
                if k in ('batch', 'size'):
                    continue
                mlflow.log_metric('validation_' + k, logs[k])
        callbacks.append(tf.keras.callbacks.LambdaCallback(on_batch_end=log_metrics_batch,
                                                           on_test_batch_end=log_metrics_validate))

        temp_dir = tempfile.mkdtemp()
        fname = os.path.join(temp_dir, 'config.yaml')
        with open(fname, 'w') as f:
            f.write(config.export())
        mlflow.log_artifact(fname)
        os.remove(fname)

        if config.mlflow_checkpoint_freq():
            class MLFlowCheckpointCallback(tf.keras.callbacks.Callback):
                def on_train_batch_end(self, batch, _):
                    if batch % config.mlflow_checkpoint_freq() == 0:
                        filename = os.path.join(temp_dir, '%d.h5' % (batch))
                        self.model.save(filename, save_format='h5')
                        if config.mlflow_checkpoint_latest():
                            old = filename
                            filename = os.path.join(temp_dir, 'latest.h5')
                            os.rename(old, filename)
                        mlflow.log_artifact(filename, 'checkpoints')
                        os.remove(filename)
            callbacks.append(MLFlowCheckpointCallback())

    try:
        history = model.fit(ds,
                            callbacks=callbacks,
                            validation_data=validation,
                            validation_steps=training_spec.validation.steps if training_spec.validation else None,
                            steps_per_epoch=training_spec.steps)
        if config.mlflow_enabled():
            model_path = os.path.join(temp_dir, 'final_model.h5') # TODO: get this out of the config file?
            model.save(model_path, save_format='h5')
            mlflow.log_artifact(model_path)
            os.remove(model_path)
    except:
        if config.mlflow_enabled():
            mlflow.end_run('FAILED')
        raise
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)

    if config.mlflow_enabled():
        mlflow.end_run()

    return model, history
