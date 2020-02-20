"""
Configuration options specific to machine learning.
"""

class ValidationSet:#pylint:disable=too-few-public-methods
    """
    Specifies the images and labels in a validation set.
    """
    def __init__(self, images=None, labels=None, from_training=False, steps=1000):
        """
        Uses the specified `delta.imagery.sources.image_set.ImageSet`s images and labels.

        If `from_training` is `True`, instead takes samples from the training set
        before they are used for training.

        The number of samples to use for validation is set by `steps`.
        """
        self.images = images
        self.labels = labels
        self.from_training = from_training
        self.steps = steps

class TrainingSpec:#pylint:disable=too-few-public-methods,too-many-arguments,dangerous-default-value
    """
    Options used in training by `delta.ml.train.train`.
    """
    def __init__(self, batch_size, epochs, loss_function, validation=None, steps=None,
                 metrics=['accuracy'], chunk_stride=1, optimizer='adam', experiment_name='Default'):
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_function = loss_function
        self.validation = validation
        self.steps = steps
        self.metrics = metrics
        self.chunk_stride = chunk_stride
        self.optimizer = optimizer
        self.experiment = experiment_name
