#pylint: disable=R0915,R0914,R0912,R0201,W0212,W0613

import mlflow
import mlflow.tensorflow
import tensorflow as tf

class Experiment:

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


    def train(self, model, train_dataset, num_epochs=70, steps_per_epoch=2024, validation_data=None, log_model=False):
        assert model is not None
        assert train_dataset is not None

        mlflow.log_param('num_epochs', num_epochs)
        mlflow.log_param('model summary', model.summary())

        model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
        estimator = tf.keras.estimator.model_to_estimator(keras_model=model)

        def input_fn():
            return train_dataset.repeat()
        estimator.train(input_fn)
        assert model is not None
#         history = model.fit(train_dataset, epochs=num_epochs,
#                         steps_per_epoch=steps_per_epoch,
#                         validation_data=validation_data)

#         for i in range(num_epochs):
#             mlflow.log_metric('loss', history.history['loss'][i])
#             mlflow.log_metric('acc',  history.history['acc' ][i])
#         ### end for
        if log_model:
            model.save('model.h5')
            mlflow.log_artifact('model.h5')
        ### end log_model

#         return history
#         return None
    ### end train

    def train_estimator(self, model, train_dataset_fn, num_epochs=70, steps_per_epoch=2024,
                        validation_data=None, log_model=False, num_gpus=1):
        """Alternate call that uses the TF Estimator interface to run on multiple GPUs"""
        assert model is not None
        assert train_dataset_fn is not None

        mlflow.log_param('num_epochs', num_epochs)
        mlflow.log_param('model summary', model.summary())

        #model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001) # TODO
        model.compile(optimizer=optimizer, loss='mean_squared_logarithmic_error', metrics=['accuracy'])

        assert model is not None

        # Set up multi-GPU strategy
        tf_config = tf.estimator.RunConfig(
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
            keras_model=model, config=tf_config)#, model_dir=config_values['ml']['model_folder'])


        # TODO: Use separate validate dataset!
#         result =
        tf.estimator.train_and_evaluate(
            keras_estimator,
            train_spec=tf.estimator.TrainSpec(input_fn=train_dataset_fn),
            eval_spec=tf.estimator.EvalSpec(input_fn=train_dataset_fn))

#         return None # In v1.12 the result is undefined for distributed training!
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

    def test(self, model, test_data, test_labels):
        assert model is not None
        assert test_data is not None
        assert test_labels is not None
        assert isinstance(test_data, type(test_labels))

        scores = model.evaluate(test_data, test_labels)
        return scores
    ### end def test

    def load_model(self, src):
        raise NotImplementedError('loading models is not yet implemented')

    def log_parameters(self, params):
#         assert type(params) is dict
        assert isinstance(params, dict)
        for k in params.keys():
            mlflow.log_param(k,params[k])
        ### end for
    ### end log_parameters

    def log_training_set(self, dataset):
        import sys
        raise NotImplementedError('%s is not yet implemented' %(sys._getframe().f_code.co_name,))

    def log_testing_set(self, dataset):
        import sys
        raise NotImplementedError('%s is not yet implemented' %(sys._getframe().f_code.co_name,))

    def log_validation_set(self, dataset):
        import sys
        raise NotImplementedError('%s is not yet implemented' %(sys._getframe().f_code.co_name,))

    def log_dataset(self, prefix, dataset):
        import sys
        raise NotImplementedError('%s is not yet implemented' %(sys._getframe().f_code.co_name,))
