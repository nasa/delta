import os.path
import tensorflow as tf
import mlflow

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
        devs = [x.name for x in tf.config.experimental.list_logical_devices('CPU')]
    else:
        devs = [x.name for x in tf.config.experimental.list_logical_devices('GPU')]
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

def _prep_datasets(ids, tc, chunk_size):
    ds = ids.dataset()
    ds = ds.batch(tc.batch_size)
    if tc.validation.from_training:
        validation = ds.take(tc.validation.steps)
        ds = ds.skip(tc.validation.steps)
    else:
        vimg = tc.validation.images
        vlabel = tc.validation.labels
        if not vimg or not vlabel:
            validation = None
        else:
            vimagery = ImageryDataset(vimg, vlabel, chunk_size, tc.chunk_stride)
            validation = vimagery.dataset().batch(tc.batch_size).take(tc.validation.steps)
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
    mlflow.log_param('Steps', training_spec.steps)
    mlflow.log_param('Loss Function', training_spec.loss_function)
    mlflow.log_param('Epochs', training_spec.epochs)
    mlflow.log_param('Batch Size', training_spec.batch_size)
    mlflow.log_param('Model Layers', len(model.layers))

def train(model_fn, dataset, training_spec, experiment_name=None):
    '''
    Trains the specified model given the images, corresponding labels, and training specification.
    '''
    with _strategy(_devices(config.gpus())).scope():
        model = model_fn()
        assert isinstance(model, tf.keras.models.Model),\
               "Model is not a Tensorflow Keras model"
        model.compile(optimizer='adam', loss=training_spec.loss_function, metrics=['accuracy'])

    input_shape = model.layers[0].input_shape
    assert len(input_shape) == 4, 'Input to network is wrong shape.'
    assert input_shape[0] is None, 'Input is not batched.'
    assert input_shape[1] == input_shape[2], 'Input to network is not chunked'
    chunk_size = input_shape[1]

    assert input_shape[3] == dataset.num_bands(), 'Number of bands in model does not match data.'

    (ds, validation) = _prep_datasets(dataset, training_spec, chunk_size)

    callbacks = []
    if config.tb_enabled():
        cb = tf.keras.callbacks.TensorBoard(log_dir=config.tb_dir(),
                                            update_freq=1000,
                                           )
        callbacks.append(cb)
    if config.checkpoint_dir():
        cb = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(config.checkpoint_dir(),
                                                                      'model.ckpt.h5'),
                                                monitor='val_loss',
                                                verbose=0,
                                                save_freq=1000,
                                                save_best_only=False,
                                                )
        callbacks.append(cb)

    if config.mlflow_enabled():
        mlflow.set_tracking_uri(config.mlflow_uri())
        mlflow.start_run(experiment_id=experiment_name)
        _log_mlflow_params(model, dataset, training_spec)

    try:
        history = model.fit(ds,
                            callbacks=callbacks,
                            validation_data=validation,
                            validation_steps=training_spec.validation.steps)
    except:
        if config.mlflow_enabled():
            mlflow.end_run('FAILED')
        raise
    if config.mlflow_enabled():
        mlflow.end_run()

    return model, history
