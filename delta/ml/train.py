#pylint: disable=no-self-use,unused-argument
import mlflow
import mlflow.tensorflow
import tensorflow as tf

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
        assert len(devs) >= num_gpus, "Requested %d GPUs with only %d available." % (num_gpus, len(devs))
        devs = devs[:num_gpus]
    ### end if num_gpus < 1
    return devs
### end get_devices


def get_distribution_strategy(devices):
    '''Given a list of TensorFlow Logical Devices, returns a distribution strategy.'''
    strategy = None
    if len(devices) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device=devices[0])
    else:
        strategy = tf.distribute.MirroredStrategy(devices=devices)
    return strategy


class Experiment:
    """TODO"""

    def __init__(self, tracking_uri, experiment_name, output_dir='./', loss_fn='mean_squared_error'):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.loss_fn = loss_fn

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        mlflow.log_param('output_dir', self.output_dir)

    ### end __init__

    def __del__(self):
        mlflow.end_run()
    ### end __del__

    def train_keras(self, model_fn, train_dataset_fn, num_epochs=70,
                    validation_data=None, num_gpus=1):
        """
        Call that uses the Keras interface to train a network.

        Arguments

        model_fn -- A zero-argument function that constructs the neural network to be trained. model_fn() should return a Keras Model
        train_dataset_fn -- A zero-argument function that constructs the dataset as a Tensorflow Dataset object.
        num_epochs -- The number of epochs to train the network for.  Default value is 70
        validation_data -- The data used to validate the network.  Default None.
        num_gpus -- The number of GPUs used to train the network.  If GPU


        """
        assert model_fn is not None, "No model function supplied."
        assert train_dataset_fn is not None, "No training dataset function supplied."
        assert num_gpus > -1, "Number of GPUs is negative."

        devs = get_devices(num_gpus)
        strategy = get_distribution_strategy(devs)
        with strategy.scope():
            model = model_fn()
            assert isinstance(model, tf.keras.models.Model), "Model is not a Tensorflow Keras model"
            model.compile(optimizer='adam', loss=self.loss_fn, metrics=['accuracy'])

        history = model.fit(train_dataset_fn(), 
                            epochs=num_epochs,
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

    def load_model(self, src):
        """TODO"""
        raise NotImplementedError('loading models is not yet implemented')

    def log_parameters(self, params):
        """
        Takes a dictionary of parameters and logs each named parameter.
        """
        assert isinstance(params, dict)
        for k in params.keys():
            mlflow.log_param(k,params[k])
        ### end for
    ### end log_parameters
