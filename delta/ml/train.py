#pylint: disable=no-self-use,unused-argument
import mlflow
import mlflow.tensorflow
import tensorflow as tf

def get_devices(num_gpus):
    if num_gpus < 1:
        return [x.name for x in tf.config.experimental.list_logical_devices('CPU')]
    ### end if num_gpus < 1
    devs = [x.name for x in tf.config.experimental.list_logical_devices('GPU')]
    return devs[:num_gpus]
### end get_devices


def train(model, train_dataset_fn, test_dataset_fn=None, model_folder=None, num_gpus=1,
          skip_train=False):
    """Plain training function without mlflow stuff.
       Converts from the input Keras model to an Estimator model.
       If skip_train is set then it will convert but not do any training.
    """

    assert model is not None
    assert train_dataset_fn is not None

    # Save a checkpoint file every 10 minutes.
    # - When restarting from a checkpoint, TF does not remember where we were in the
    #   input dataset so it will not train "evenly" but hopefully if the randomization
    #   is good enough and/or there are multiple epochs this won't matter.
    CHECKPOINT_SPACING_SECONDS = 10 * 60

    # Set up multi-GPU strategy
    tf_config = tf.estimator.RunConfig(
        save_checkpoints_secs=CHECKPOINT_SPACING_SECONDS,
        experimental_distribute=tf.contrib.distribute.DistributeConfig(
                train_distribute=tf.contrib.distribute.MirroredStrategy( #pylint: disable=C0330
                    num_gpus_per_worker=num_gpus,
                    ),
                eval_distribute=tf.contrib.distribute.MirroredStrategy( #pylint: disable=C0330
                    num_gpus_per_worker=num_gpus,
                    )))
    #tf_config = tf.estimator.RunConfig() # DEBUG: Force single GPU

    # Convert from Keras to Estimator
    print('Calling model_to_estimator...')
    keras_estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, config=tf_config, model_dir=model_folder)
    if skip_train:
        return keras_estimator

    if test_dataset_fn is None:
        # It appears this is the only way to skip the evaluation step
        eval_spec=tf.estimator.EvalSpec(input_fn=train_dataset_fn, steps=None)
    else:
        eval_spec=tf.estimator.EvalSpec(input_fn=test_dataset_fn)
    print('Calling train_and_evaluate...')
    tf.estimator.train_and_evaluate( #pylint: disable=W0612
        keras_estimator,
        train_spec=tf.estimator.TrainSpec(input_fn=train_dataset_fn),
        eval_spec=eval_spec)

    # keras_estimator.evaluate(input_fn=test_dataset_fn) # TODO Run this?
    return keras_estimator

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
                    validation_data=None, log_model=False, num_gpus=1):
        """Call that uses the Keras interface, only works on a single GPU"""
        assert model_fn is not None
        assert train_dataset_fn is not None

        devs = get_devices(num_gpus)
        if len(devs) == 1:
            strategy = tf.distribute.OneDeviceStrategy(device=devs[0])
        else:
            strategy = tf.distribute.MirroredStrategy(devices=devs)
        with strategy.scope():
            model = model_fn()
            model.compile(optimizer='adam', loss=self.loss_fn, metrics=['accuracy'])

        history = model.fit(train_dataset_fn(), epochs=num_epochs,
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
        """TODO"""
        assert isinstance(params, dict)
        for k in params.keys():
            mlflow.log_param(k,params[k])
        ### end for
    ### end log_parameters
