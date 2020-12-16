# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Configuration options specific to machine learning.
"""
# Please do not put any tensorflow imports in this file as it will greatly slow loading
# when tensorflow isn't needed
import os.path

import appdirs
import pkg_resources
import yaml

from delta.imagery.imagery_config import ImageSet, ImageSetConfig, load_images_labels
import delta.config as config

class ValidationSet:#pylint:disable=too-few-public-methods
    """
    Specifies the images and labels in a validation set.
    """
    def __init__(self, images=None, labels=None, from_training=False, steps=1000):
        """
        Uses the specified `delta.imagery.imagery_config.ImageSet`s images and labels.

        If `from_training` is `True`, instead takes samples from the training set
        before they are used for training.

        The number of samples to use for validation is set by `steps`.
        """
        self.images = images
        self.labels = labels
        self.from_training = from_training
        self.steps = steps

class TrainingSpec:#pylint:disable=too-few-public-methods,too-many-arguments
    """
    Options used in training by `delta.ml.train.train`.
    """
    def __init__(self, batch_size, epochs, loss, metrics, validation=None, steps=None,
                 chunk_stride=None, optimizer='Adam'):
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.validation = validation
        self.steps = steps
        self.metrics = metrics
        self.chunk_stride = chunk_stride
        self.optimizer = optimizer

class NetworkModelConfig(config.DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self.register_field('yaml_file', str, 'yaml_file', config.validate_path,
                            'A YAML file describing the network to train.')
        self.register_field('params', dict, None, None, None)
        self.register_field('layers', list, None, None, None)

    # overwrite model entirely if updated (don't want combined layers from multiple files)
    def _load_dict(self, d : dict, base_dir):
        super()._load_dict(d, base_dir)
        if 'yaml_file' in d:
            self._config_dict['layers'] = None
        elif 'layers' in d:
            self._config_dict['yaml_file'] = None
        if 'yaml_file' in d and 'layers' in d and d['yaml_file'] is not None and d['layers'] is not None:
            raise ValueError('Specified both yaml file and layers in model.')
        if 'yaml_file' in d and d['yaml_file'] is not None:
            yaml_file = d['yaml_file']
            resource = os.path.join('config', yaml_file)
            if not os.path.exists(yaml_file) and pkg_resources.resource_exists('delta', resource):
                yaml_file = pkg_resources.resource_filename('delta', resource)
            if not os.path.exists(yaml_file):
                raise ValueError('Model yaml_file does not exist: ' + yaml_file)
            with open(yaml_file, 'r') as f:
                self._config_dict.update(yaml.safe_load(f))

def validate_size(size, _):
    if size is None:
        return size
    assert len(size) == 2, 'Size must be tuple.'
    assert isinstance(size[0], int) and isinstance(size[1], int), 'Size must be integer.'
    assert size[0] > 0 and size[1] > 0, 'Size must be positive.'
    return size

class NetworkConfig(config.DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self.register_component(NetworkModelConfig(), 'model')

    def setup_arg_parser(self, parser, components = None) -> None:
        group = parser.add_argument_group('Network')
        super().setup_arg_parser(group, components)

class ValidationConfig(config.DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self.register_field('steps', int, 'steps', config.validate_positive,
                            'If from training, validate for this many steps.')
        self.register_field('from_training', bool, 'from_training', None,
                            'Take validation data from training data.')
        self.register_component(ImageSetConfig(), 'images', '__image_comp')
        self.register_component(ImageSetConfig(), 'labels', '__label_comp')
        self.__images = None
        self.__labels = None

    def reset(self):
        super().reset()
        self.__images = None
        self.__labels = None

    def images(self) -> ImageSet:
        """
        Returns the training images.
        """
        if self.__images is None:
            (self.__images, self.__labels) = load_images_labels(self._components['images'],
                                                                self._components['labels'],
                                                                config.config.dataset.classes)
        return self.__images

    def labels(self) -> ImageSet:
        """
        Returns the label images.
        """
        if self.__labels is None:
            (self.__images, self.__labels) = load_images_labels(self._components['images'],
                                                                self._components['labels'],
                                                                config.config.dataset.classes)
        return self.__labels

def _validate_chunk_stride(stride, _):
    if stride is None:
        return None
    if isinstance(stride, int):
        stride = (stride, stride)
    assert len(stride) == 2, 'Stride must have two components.'
    assert isinstance(stride[0], int) and isinstance(stride[1], int), 'Stride must be integer.'
    assert stride[0] > 0 and stride[1] > 0, 'Stride must be positive.'
    return stride

class TrainingConfig(config.DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self.register_field('chunk_stride', (list, int, None), None, _validate_chunk_stride,
                            'Pixels to skip when iterating over chunks. A value of 1 means to take every chunk.')
        self.register_field('epochs', int, None, config.validate_positive,
                            'Number of times to repeat training on the dataset.')
        self.register_field('batch_size', int, None, config.validate_positive,
                            'Features to group into each training batch.')
        self.register_field('loss', (str, dict), None, None, 'Keras loss function.')
        self.register_field('metrics', list, None, None, 'List of metrics to apply.')
        self.register_field('steps', int, None, config.validate_non_negative, 'Batches to train per epoch.')
        self.register_field('optimizer', (str, dict), None, None, 'Keras optimizer to use.')
        self.register_field('callbacks', list, 'callbacks', None, 'Callbacks used to modify training')
        self.register_arg('epochs', '--epochs')
        self.register_arg('batch_size', '--batch-size')
        self.register_arg('steps', '--steps')
        self.register_component(ValidationConfig(), 'validation')
        self.register_component(NetworkConfig(), 'network')
        self.__training = None

    def setup_arg_parser(self, parser, components = None) -> None:
        group = parser.add_argument_group('Training')
        super().setup_arg_parser(group, components)

    def spec(self) -> TrainingSpec:
        """
        Returns the options configuring training.
        """
        if not self.__training:
            from_training = self._components['validation'].from_training()
            vsteps = self._components['validation'].steps()
            (vimg, vlabels) = (None, None)
            if not from_training:
                (vimg, vlabels) = (self._components['validation'].images(), self._components['validation'].labels())
            validation = ValidationSet(vimg, vlabels, from_training, vsteps)
            self.__training = TrainingSpec(batch_size=self._config_dict['batch_size'],
                                           epochs=self._config_dict['epochs'],
                                           loss=self._config_dict['loss'],
                                           metrics=self._config_dict['metrics'],
                                           validation=validation,
                                           steps=self._config_dict['steps'],
                                           chunk_stride=self._config_dict['chunk_stride'],
                                           optimizer=self._config_dict['optimizer'])
        return self.__training


class MLFlowCheckpointsConfig(config.DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self.register_field('frequency', int, 'frequency', None,
                            'Frequency in batches to store neural network checkpoints.')
        self.register_field('only_save_latest', bool, 'only_save_latest', None,
                            'If true, only keep the most recent checkpoint.')

class MLFlowConfig(config.DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self.register_field('enabled', bool, 'enabled', None, 'Enable MLFlow.')
        self.register_field('uri', str, None, None, 'URI to store MLFlow data.')
        self.register_field('frequency', int, 'frequency', config.validate_positive,
                            'Frequency to store metrics.')
        self.register_field('experiment_name', str, 'experiment', None, 'Experiment name in MLFlow.')

        self.register_arg('enabled', '--disable-mlflow', action='store_const', const=False, type=None)
        self.register_arg('enabled', '--enable-mlflow', action='store_const', const=True, type=None)
        self.register_component(MLFlowCheckpointsConfig(), 'checkpoints')

    def uri(self) -> str:
        """
        Returns the URI for MLFlow to store data.
        """
        uri = self._config_dict['uri']
        if uri == 'default':
            uri = 'file://' + os.path.join(appdirs.AppDirs('delta', 'nasa').user_data_dir, 'mlflow')
        return uri

class TensorboardConfig(config.DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self.register_field('enabled', bool, 'enabled', None, 'Enable Tensorboard.')
        self.register_field('dir', str, None, None, 'Directory to store Tensorboard data.')

    def dir(self) -> str:
        """
        Returns the directory for tensorboard to store to.
        """
        tbd = self._config_dict['dir']
        if tbd == 'default':
            tbd = os.path.join(appdirs.AppDirs('delta', 'nasa').user_data_dir, 'tensorboard')
        return tbd

def register():
    """
    Registers machine learning config options with the global config manager.

    The arguments enable command line arguments for different components.
    """
    config.config.general.register_field('gpus', int, 'gpus', None, 'Number of gpus to use.')
    config.config.general.register_arg('gpus', '--gpus')

    config.config.register_component(TrainingConfig(), 'train')
    config.config.register_component(MLFlowConfig(), 'mlflow')
    config.config.register_component(TensorboardConfig(), 'tensorboard')
