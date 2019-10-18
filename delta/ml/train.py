#pylint: disable=no-self-use,unused-argument
import mlflow
import mlflow.tensorflow
import tensorflow as tf

def train(model, train_dataset_fn, test_dataset_fn=None, model_folder=None, num_gpus=1):
    """Plain training function without mlflow stuff"""

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
    keras_estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, config=tf_config, model_dir=model_folder)

    if test_dataset_fn is None:
        # It appears this is the only way to skip the evaluation step
        eval_spec=tf.estimator.EvalSpec(input_fn=train_dataset_fn, steps=None)
    else:
        eval_spec=tf.estimator.EvalSpec(input_fn=test_dataset_fn)
    tf.estimator.train_and_evaluate( #pylint: disable=W0612
        keras_estimator,
        train_spec=tf.estimator.TrainSpec(input_fn=train_dataset_fn),
        eval_spec=eval_spec)

    # keras_estimator.evaluate(input_fn=test_dataset_fn) # TODO Run this?
    return keras_estimator, tf_config.eval_distribute.scope()

class Experiment:
    """TODO"""

    def __init__(self, tracking_uri, experiment_name, output_dir='./'):
        self.experiment_name = experiment_name
        self.output_dir = output_dir

        mlflow.set_tracking_uri(tracking_uri)
#         client = mlflow.tracking.MlflowClient(tracking_uri)
#         exp = client.get_experiment_by_name(experiment_name)
#         experiment_id = None
#         if exp is None:
#             experiment_id = mlflow.create_experiment(experiment_name)
#         else:
        mlflow.set_experiment(experiment_name)
#         run = mlflow.start_run()
        mlflow.start_run()
        mlflow.log_param('output_dir', self.output_dir)

    ### end __init__

    def __del__(self):
        mlflow.end_run()
    ### end __del__

    def train(self, model, train_dataset_fn, test_dataset_fn=None, model_folder=None, #pylint: disable=R0913
              num_epochs=70, steps_per_epoch=2024,
              validation_data=None, log_model=False, num_gpus=1):
        """Train call that uses the TF Estimator interface to run on multiple GPUs"""

        mlflow.log_param('num_epochs', num_epochs)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001) # TODO
        model.compile(optimizer=optimizer, loss='mean_squared_logarithmic_error', metrics=['accuracy'])

        mlflow.log_param('model summary', model.summary())
        # Call the lower level estimator train function
        estimator_model, distribution_scope = train(model, train_dataset_fn, test_dataset_fn, model_folder, num_gpus)
        return estimator_model, distribution_scope

        # TODO: Record the output from the Estimator!

        #for i in range(num_epochs):
        #    mlflow.log_metric('loss', history.history['loss'][i])
        #    mlflow.log_metric('acc',  history.history['acc' ][i])
        #### end for
        #if log_model:
        #    model.save('model.h5')
        #    mlflow.log_artifact('model.h5')
        ### end log_model
        #return history
    ### end train


    def train_keras(self, model, train_dataset_fn,
                    num_epochs=70, steps_per_epoch=2024,
                    validation_data=None, log_model=False, num_gpus=1):
        """Call that uses the Keras interface, only works on a single GPU"""
        assert model is not None
        assert train_dataset_fn is not None

        mlflow.log_param('num_epochs', num_epochs)
        mlflow.log_param('model summary', model.summary())

        model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])

        assert model is not None

        history = model.fit(train_dataset_fn(), epochs=num_epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=validation_data)

        for i in range(num_epochs):
            mlflow.log_metric('loss', history.history['loss'][i])
            mlflow.log_metric('acc',  history.history['acc' ][i])
        ### end for
        if log_model:
            model.save('model.h5')
            mlflow.log_artifact('model.h5')
        ## end log_model

        return model
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
