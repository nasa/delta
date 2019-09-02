import mlflow
import mlflow.tensorflow
import os.path

class Experiment(object):

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
        experiment_id = mlflow.set_experiment(experiment_name)
        run = mlflow.start_run()
        mlflow.log_param('output_dir', self.output_dir)
        
    ### end __init__

    def __del__(self):
        mlflow.end_run()
    ### end __del__

#     def train(self, model, train_data, train_labels, num_epochs=70, batch_size=2048, validation_data=None, log_model=False):
#         assert(model is not None)
#         assert(train_data is not None)
#         assert(train_labels is not None)
#         assert(type(train_data) == type(train_labels))
# 
#         mlflow.log_param('num_epochs', num_epochs)
#         mlflow.log_param('batch_size', batch_size)
#         mlflow.log_param('training_samples', train_data.shape[0])
#         mlflow.log_param('test_samples', train_labels.shape[0])
#         mlflow.log_param('model summary', model.summary())
# 
#         model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
#         history = model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size,
#                         validation_data=validation_data)
# 
#         for i in range(num_epochs):
#             mlflow.log_metric('loss', history.history['loss'][i])
#             mlflow.log_metric('acc',  history.history['acc' ][i])
#         ### end for
#         if log_model:
#             model.save('model.h5')
#             mlflow.log_artifact('model.h5')
#         ### end log_model
# 
#         return history
#     ### end train

    def train(self, model, train_dataset, num_epochs=70, steps_per_epoch=2024, validation_data=None, log_model=False):
        assert(model is not None)
        assert(train_dataset is not None)

        mlflow.log_param('num_epochs', num_epochs)
        mlflow.log_param('model summary', model.summary())

        model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
        assert(model is not None)
        history = model.fit(train_dataset, epochs=num_epochs, 
                        steps_per_epoch=steps_per_epoch,
                        validation_data=validation_data)

        for i in range(num_epochs):
            mlflow.log_metric('loss', history.history['loss'][i])
            mlflow.log_metric('acc',  history.history['acc' ][i])
        ### end for
        if log_model:
            model.save('model.h5')
            mlflow.log_artifact('model.h5')
        ### end log_model

        return history
    ### end train

    def test(self, model, test_data, test_labels):
        assert(model is not None)
        assert(test_data is not None)
        assert(test_labels is not None)
        assert(type(test_data) == type(test_labels))

        scores = model.evaluate(test_data, test_lables)
        pass

    def load_model(self, src):
        raise NotImplementedError('loading models is not yet implemented')

    def log_parameters(self, params):
        assert(type(params) is dict)
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
