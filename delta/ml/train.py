'''
Functions for training neural networks using the tensorflow 2.0 Keras api.  Also
provides Experiment class for tracking network parameters and performance.
'''

#pylint: disable=no-self-use,unused-argument,too-many-arguments,unexpected-keyword-arg
import os.path
import tensorflow as tf
import mlflow
import mlflow.tensorflow

from delta.config import config


def get_devices(num_gpus):
    '''
    Takes a number of GPUs and returns a list of TensorFlow LogicalDevices.

    Arguments

    num_gpus -- Number of GPUs to use.  If zero, will use all CPUs available.
    '''
    assert num_gpus > -1, "Requested negative GPUs"
    devs = None
    if num_gpus < 1:
        devs = [x.name for x in tf.config.experimental.list_logical_devices('CPU')]
    else:
        devs = [x.name for x in tf.config.experimental.list_logical_devices('GPU')]
        assert len(devs) >= num_gpus,\
               "Requested %d GPUs with only %d available." % (num_gpus, len(devs))
        devs = devs[:num_gpus]
    return devs

def get_distribution_strategy(devices):
    '''Given a list of TensorFlow Logical Devices, returns a distribution strategy.'''
    strategy = None
    if len(devices) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device=devices[0])
    else:
        strategy = tf.distribute.MirroredStrategy(devices=devices)
    return strategy

def train(model_fn, train_dataset, optimizer='adam', loss_fn='mse', callbacks=None,
          validation_data=None, num_gpus=0):
    '''
    Trains a model constructed by model_fn on datasaet train_dataset
    '''

    assert model_fn is not None, "No model function supplied."
    assert train_dataset is not None, "No training dataset supplied."
    assert num_gpus > -1, "Number of GPUs is negative."

    devs = get_devices(num_gpus)
    strategy = get_distribution_strategy(devs)
    with strategy.scope():
        model = model_fn()
        assert isinstance(model, tf.keras.models.Model), "Model is not a Tensorflow Keras model"
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    ### end with

    history = model.fit(train_dataset,
                        callbacks=callbacks,
                        validation_data=validation_data)

    return model, history
### end train

def load_keras_model(file_path, num_gpus=1):
    """Loads a saved keras model from disk ready to use multiple_GPUs.
       This function is NOT compatible with the Experiment class!"""
    devices  = get_devices(num_gpus)
    strategy = get_distribution_strategy(devices)
    with strategy.scope():
        model = tf.keras.models.load_model(file_path)
        return model


class Experiment:
    """TODO"""

    def __init__(self, experiment_name,
                 loss_fn='mean_squared_error', save_freq=100000):
        self.experiment_name = experiment_name
        self.loss_fn = loss_fn
        self.save_freq = save_freq # Checkpoint the model every save_freq samples.

        mlflow.set_tracking_uri(config.mlflow_dir())
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()

    ### end __init__

    def __del__(self):
        mlflow.end_run()
    ### end __del__

    def train_keras(self, model_fn, train_dataset,
                    validation_data=None, num_gpus=1):
        """
        Call that uses the Keras interface to train a network.

        Arguments

        model_fn -- A zero-argument function that constructs the neural network to be
                    trained. model_fn() should return a Keras Model
        train_dataset -- A Tensorflow Dataset object. All data is evaluted.
        validation_data -- The data used to validate the network.  Default None.
        num_gpus -- The number of GPUs used to train the network.  If GPU


        """
        assert model_fn is not None, "No model function supplied."
        assert train_dataset is not None, "No training dataset supplied."
        assert num_gpus > -1, "Number of GPUs is negative."

        devs = get_devices(num_gpus)
        strategy = get_distribution_strategy(devs)
        with strategy.scope():
            model = model_fn()
            assert isinstance(model, tf.keras.models.Model),\
                   "Model is not a Tensorflow Keras model"
            model.compile(optimizer='adam', loss=self.loss_fn, metrics=['accuracy'])
        ### end with

        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=config.tb_dir(),
                update_freq=self.save_freq,
                )
        ]
        if config.checkpoint_dir():
            cb = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(config.checkpoint_dir(),
                                                                          'model.ckpt.h5'),
                                                    monitor='val_loss',
                                                    verbose=0,
                                                    save_freq=self.save_freq,
                                                    save_best_only=False,
                                                    )
            callbacks.append(cb)

        history = model.fit(train_dataset,
                            callbacks=callbacks,
                            validation_data=validation_data)

        return model, history
        ### end train

    def test(self, model, test_data, test_labels):
        """Evaluate the model on the input dataset"""
        assert model is not None
        assert test_data is not None
        assert test_labels is not None
        assert isinstance(test_data, type(test_labels))

        # This probably won't work with multiple GPUs in 1.12
        scores = model.evaluate(test_data, test_labels)
        return scores
    ### end def test

    #def load_model(self, src):
    #    """TODO"""
    #    raise NotImplementedError('loading models is not yet implemented')

    def log_parameters(self, params):
        """
        Takes a dictionary of parameters and logs each named parameter.
        """
        assert isinstance(params, dict)
        for k in params.keys():
            mlflow.log_param(k, params[k])
        ### end for
    ### end log_parameters
