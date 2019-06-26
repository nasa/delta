import mlflow

def train(model, train_data, train_labels, num_epochs=70, batch_size=2048, validation_data=None):
    mlflow.log_param('num_epochs', num_epochs)
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param('training_samples', train_data.shape[0])
    mlflow.log_param('test_samples', train_labels.shape[0])

    model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
    history = model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size,
                        validation_data=validation_data)

    for i in range(num_epochs):
        mlflow.log_metric('loss', history.history['loss'][i])
        mlflow.log_metric('acc',  history.history['acc' ][i])

    return history
