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

#pylint: disable=no-member

import os
import os.path

import numpy as np
import yaml
import pkg_resources
import appdirs
from delta.imagery import disk_folder_cache
from delta.imagery.sources import image_set
from delta.ml import ml_config

__DEFAULT_EXTENSIONS = {'tiff' : '.tiff',
                        'worldview' : '.zip',
                        'landsat' : '.zip',
                        'npy' : '.npy'}
__DEFAULT_SCALE_FACTORS = {'tiff' : 1024.0,
                           'worldview' : 2048.0,
                           'landsat' : 120.0,
                           'npy' : None}
def __extension(conf):
    if conf['extension'] == 'default':
        return __DEFAULT_EXTENSIONS.get(conf['type'])
    return conf['extension']
def __scale_factor(image_comp):
    f = image_comp.preprocess.scale_factor()
    if f == 'default':
        return __DEFAULT_SCALE_FACTORS.get(image_comp.type())
    try:
        return float(f)
    except ValueError:
        raise ValueError('Scale factor is %s, must be a float.' % (f))

def __find_images(conf, matching_images=None, matching_conf=None):
    '''
    Find the images specified by a given configuration, returning a list of images.
    If matching_images and matching_conf are specified, we find the labels matching these images.
    '''
    images = []
    if (conf['files'] is None) != (conf['file_list'] is None) != (conf['directory'] is None):
        raise  ValueError('''Too many image specification methods used.\n
                             Choose one of "files", "file_list" and "directory" when indicating 
                             file locations.''')
    if conf['type'] not in __DEFAULT_EXTENSIONS:
        raise ValueError('Unexpected image type %s.' % (conf['type']))

    if conf['files']:
        images = conf['files']
    elif conf['file_list']:
        with open(conf['file_list'], 'r') as f:
            for line in f:
                images.append(line)
    elif conf['directory']:
        extension = __extension(conf)
        if not os.path.exists(conf['directory']):
            raise ValueError('Supplied images directory %s does not exist.' % (conf['directory']))
        if matching_images is None:
            for root, _, filenames in os.walk(conf['directory']):
                for filename in filenames:
                    if filename.endswith(extension):
                        images.append(os.path.join(root, filename))
        else:
            # find matching labels
            for m in matching_images:
                rel_path   = os.path.relpath(m, matching_conf['directory'])
                label_path = os.path.join(conf['directory'], rel_path)
                images.append(os.path.splitext(label_path)[0] + extension)

    for img in images:
        if not os.path.exists(img):
            raise ValueError('Image file %s does not exist.' % (img))
    return images

def __preprocess_function(image_comp):
    if not image_comp.preprocess.enabled():
        return None
    f = __scale_factor(image_comp)
    if f is None:
        return None
    return lambda data, _, dummy: data / np.float32(f)

def _config_to_image_label_sets(images_comp, labels_comp):
    '''
    Takes two configuration subsections and returns (image set, label set)
    '''
    images_dict = images_comp._config_dict #pylint:disable=protected-access
    labels_dict = labels_comp._config_dict #pylint:disable=protected-access
    images = __find_images(images_dict)

    if images_dict['directory']:
        if labels_dict['files'] or labels_dict['file_list']:
            raise ValueError('Image directory only supported with label directory.')
        if labels_dict['directory']:
            # remove images in same directory ending with label's extension (can have .tiff and _label.tiff in same dir)
            if os.path.realpath(labels_dict['directory']).startswith(os.path.realpath(images_dict['directory'])):
                label_extension = __extension(labels_dict)
                images = [img for img in images if not img.endswith(label_extension)]

    pre = __preprocess_function(images_comp)
    imageset = image_set.ImageSet(images, images_dict['type'], pre, images_dict['nodata_value'])

    if (labels_dict['files'] is None) and (labels_dict['file_list'] is None) and (labels_dict['directory'] is None):
        return (imageset, None)

    labels = __find_images(labels_dict, images, images_dict)

    if len(labels) != len(images):
        raise ValueError('%d images found, but %d labels found.' % (len(images), len(labels)))

    pre = __preprocess_function(labels_comp)
    return (imageset, image_set.ImageSet(labels, labels_dict['type'],
                                         pre, labels_dict['nodata_value']))

def validate_path(path, base_dir):
    path = os.path.expanduser(path)
    # make relative paths relative to this config file
    if base_dir:
        path = os.path.normpath(os.path.join(base_dir, path))
    return path

def validate_positive(num, _):
    if num <= 0:
        raise ValueError('%d is not positive' % (num))
    return num

class DeltaConfigComponent:
    """
    DELTA configuration component.

    Handles one subsection of a config file. Generally subclasses
    will want to register fields and components in the constructor,
    and possibly override setup_arg_parser and parse_args to handle
    command line options.
    """
    def __init__(self):
        """
        Constructs the component.
        """
        self._config_dict = {}
        self._components = {}
        self._fields = []
        self._validate = {}
        self._types = {}
        self._cmd_args = {}
        self._descs = {}

    def reset(self):
        """
        Resets all state in the component.
        """
        self._config_dict = {}
        for c in self._components.values():
            c.reset()

    def register_component(self, component, name : str, attr_name = None):
        """
        Register a subcomponent with a name and attribute name (access as self.attr_name)
        """
        assert name not in self._components
        self._components[name] = component
        if attr_name is None:
            attr_name = name
        setattr(self, attr_name, component)

    def _register_field(self, name : str, types, accessor = None, cmd_arg = None, validate_fn = None, desc = None):
        """
        Register a field in this component of the configuration.

        types is a single type or a tuple of valid types

        validate_fn (optional) should take two strings as input, the field's value and
        the base directory, and return what to save to the config dictionary.
        It should raise an exception if the field is invalid.
        accessor is an optional name to create an accessor function with
        """
        self._fields.append(name)
        self._validate[name] = validate_fn
        self._types[name] = types
        self._cmd_args[name] = cmd_arg
        self._descs[name] = desc
        if accessor:
            def access(self) -> types:
                return self._config_dict[name]#pylint:disable=protected-access
            access.__name__ = accessor
            access.__doc__ = desc
            setattr(self.__class__, accessor, access)

    def export(self) -> str:
        """
        Returns a YAML string of all configuration options.
        """
        return yaml.dump(self.__config_dict)

    def _set_field(self, name : str, value : str, base_dir : str):
        if name not in self._fields:
            raise ValueError('Unexpected field %s.' % (name))
        if value is not None and not isinstance(value, self._types[name]):
            raise TypeError('%s must be of type %s, is %s.' % (name, self._types[name], value))
        if self._validate[name] and value is not None:
            try:
                value = self._validate[name](value, base_dir)
            except:
                print('Value %s for %s is invalid.' % (value, name))
                raise
        self._config_dict[name] = value

    def _load_dict(self, d : dict, base_dir):
        """
        Loads the dictionary d, assuming it came from the given base_dir (for relative paths).
        """
        for (k, v) in d.items():
            if k in self._components:
                self._components[k]._load_dict(v, base_dir) #pylint:disable=protected-access
            else:
                self._set_field(k, v, base_dir)

    def setup_arg_parser(self, parser) -> None:
        """
        Adds arguments to the parser. Must overridden by child classes.
        """
        for name in self._fields:
            c = self._cmd_args[name]
            if c is None:
                continue
            parser.add_argument(c, dest=c.replace('-', '_'), required=False,
                                type=self._types[name], help=self._descs[name])

    def parse_args(self, options):
        """
        Parse options extracted from an ArgParser configured with
        `setup_arg_parser` and override the appropriate
        configuration values.
        """
        d = {}
        for name in self._fields:
            c = self._cmd_args[name]
            if c is None:
                continue
            n = c.replace('-', '_')
            if not hasattr(options, n):
                continue
            d[name] = getattr(options, n)
        self._load_dict(d, None)
        return options

class CacheConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self._register_field('dir', str, None, None, validate_path, 'Cache directory.')
        self._register_field('limit', int, None, None, validate_positive, 'Number of items to cache.')

        self._cache_manager = None

    def reset(self):
        super().reset()
        self._cache_manager = None

    def manager(self) -> disk_folder_cache.DiskCache:
        """
        Returns the disk cache object to manage the cache.
        """
        if self._cache_manager is None:
            cdir = self._config_dict['dir']
            if cdir == 'default':
                cdir = appdirs.AppDirs('delta', 'nasa').user_cache_dir
            self._cache_manager = disk_folder_cache.DiskCache(cdir, self._config_dict['limit'])
        return self._cache_manager


class GeneralConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self._register_field('gpus', int, 'gpus', '--gpus', None, 'Number of gpus to use.')
        self._register_field('threads', int, 'threads', '--threads', None, 'Number of threads to use.')
        self._register_field('block_size_mb', int, 'block_size_mb', '--block-size-mb', validate_positive,
                             'Size of an image block to load in memory at once.')
        self._register_field('interleave_images', int, 'interleave_images', None, validate_positive,
                             'Number of images to interleave at a time when training.')
        self._register_field('tile_ratio', float, 'tile_ratio', '--tile-ratio', validate_positive,
                             'Width to height ratio of blocks to load in images.')
        self.register_component(CacheConfig(), 'cache')

    def setup_arg_parser(self, parser) -> None:
        group = parser.add_argument_group('General')
        super().setup_arg_parser(group)

class ImagePreprocessConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self._register_field('enabled', bool, 'enabled', None, None, 'Turn on preprocessing.')
        self._register_field('scale_factor', (float, str), 'scale_factor', None, None, 'Image scale factor.')

class ImageSetConfig(DeltaConfigComponent):
    def __init__(self, name=None):
        super().__init__()
        self._register_field('type', str, 'type', '--' + name + '-type' if name else None, None, 'Image type.')
        self._register_field('files', list, None, None, None, 'List of image files.')
        self._register_field('file_list', list, None, '--' + name + '-file-list' if name else None,
                             validate_path, 'File listing image files.')
        self._register_field('directory', str, None, '--' + name + '-dir' if name else None,
                             validate_path, 'Directory of image files.')
        self._register_field('extension', str, None, '--' + name + '-extension' if name else None,
                             None, 'Image file extension.')
        self._register_field('nodata_value', float, None, None, None, 'Value of pixels to ignore.')
        self.register_component(ImagePreprocessConfig(), 'preprocess')
        self._name = name

    def setup_arg_parser(self, parser) -> None:
        if self._name:
            parser.add_argument("--" + self._name, dest=self._name, required=False,
                                help="Specify a single image file.")
        super().setup_arg_parser(parser)

    def parse_args(self, options):
        super().parse_args(options)
        if hasattr(options, self._name) and getattr(options, self._name) is not None:
            self._config_dict['files'] = [getattr(options, self._name)]
        return options

class DatasetConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self.register_component(ImageSetConfig('image'), 'images', '__image_comp')
        self.register_component(ImageSetConfig('label'), 'labels', '__label_comp')
        self.__images = None
        self.__labels = None

    def reset(self):
        super().reset()
        self.__images = None
        self.__labels = None

    def setup_arg_parser(self, parser) -> None:
        group = parser.add_argument_group('Dataset')
        super().setup_arg_parser(group)
        for c in self._components.values():
            c.setup_arg_parser(group)

    def parse_args(self, options):
        for c in self._components.values():
            c.parse_args(options)
        return options

    def images(self) -> image_set.ImageSet:
        """
        Returns the training images.
        """
        if self.__images is None:
            (self.__images, self.__labels) = _config_to_image_label_sets(self._components['images'],
                                                                         self._components['labels'])
        return self.__images

    def labels(self) -> image_set.ImageSet:
        """
        Returns the label images.
        """
        if self.__labels is None:
            (self.__images, self.__labels) = _config_to_image_label_sets(self._components['images'],
                                                                         self._components['labels'])
        return self.__labels


class NetworkModelConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self._register_field('yaml_file', str, 'yaml_file', None, validate_path,
                             'A YAML file describing the network to train.')
        self._register_field('params', dict, None, None, None, None)
        self._register_field('layers', list, None, None, None, None)

    # overwrite model entirely if updated (don't want combined layers from multiple files)
    def _load_dict(self, d : dict, base_dir):
        super()._load_dict(d, base_dir)
        for k in ['yaml_file', 'layers', 'params']:
            if not k in d:
                self._config_dict[k] = None

    def as_dict(self) -> dict:
        """
        Returns a dictionary representing the network model for use by `delta.ml.model_parser`.
        """
        yaml_file = self._config_dict['yaml_file']
        if yaml_file is not None:
            if self._config_dict['layers'] is not None:
                raise ValueError('Specified both yaml file and layers in model.')

            resource = os.path.join('config', yaml_file)
            if not os.path.exists(yaml_file) and pkg_resources.resource_exists('delta', resource):
                yaml_file = pkg_resources.resource_filename('delta', resource)
            if not os.path.exists(yaml_file):
                raise ValueError('Model yaml_file does not exist: ' + yaml_file)
            with open(yaml_file, 'r') as f:
                return yaml.safe_load(f)
        return self._config_dict

class NetworkConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self._register_field('chunk_size', int, 'chunk_size', '--chunk-size', validate_positive,
                             'Width of an image chunk to input to the neural network.')
        self._register_field('output_size', int, 'output_size', '--output-size', validate_positive,
                             'Width of an image chunk to output from the neural network.')
        self._register_field('classes', int, 'classes', '--classes', validate_positive, 'Number of label classes.')
        self.register_component(NetworkModelConfig(), 'model')

    def setup_arg_parser(self, parser) -> None:
        group = parser.add_argument_group('Network')
        super().setup_arg_parser(group)

class ValidationConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self._register_field('steps', int, 'steps', None, validate_positive,
                             'If from training, validate for this many steps.')
        self._register_field('from_training', bool, 'from_training', None, None,
                             'Take validation data from training data.')
        self.register_component(ImageSetConfig(), 'images')
        self.register_component(ImageSetConfig(), 'labels')
        self.__images = None
        self.__labels = None

    def reset(self):
        super().reset()
        self.__images = None
        self.__labels = None

    def images(self) -> image_set.ImageSet:
        """
        Returns the training images.
        """
        if self.__images is None:
            (self.__images, self.__labels) = _config_to_image_label_sets(self._components['images'],
                                                                         self._components['labels'])
        return self.__images

    def labels(self) -> image_set.ImageSet:
        """
        Returns the label images.
        """
        if self.__labels is None:
            (self.__images, self.__labels) = _config_to_image_label_sets(self._components['images'],
                                                                         self._components['labels'])
        return self.__labels

class TrainingConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self._register_field('chunk_stride', int, None, '--chunk-stride', validate_positive,
                             'Pixels to skip when iterating over chunks. A value of 1 means to take every chunk.')
        self._register_field('epochs', int, None, '--epochs', validate_positive,
                             'Number of times to repeat training on the dataset.')
        self._register_field('batch_size', int, None, '--batch-size', validate_positive,
                             'Features to group into each training batch.')
        self._register_field('loss_function', str, None, None, None, 'Keras loss function.')
        self._register_field('metrics', list, None, None, None, 'List of metrics to apply.')
        self._register_field('steps', int, None, '--steps', validate_positive, 'Batches to train per epoch.')
        self._register_field('experiment_name', str, None, None, None, 'Experiment name in MLFlow.')
        self._register_field('optimizer', str, None, None, None, 'Keras optimizer to use.')
        self.register_component(ValidationConfig(), 'validation')
        self.__training = None

    def setup_arg_parser(self, parser) -> None:
        group = parser.add_argument_group('Training')
        super().setup_arg_parser(group)

    def spec(self) -> ml_config.TrainingSpec:
        """
        Returns the options configuring training.
        """
        if not self.__training:
            from_training = self.validation.from_training()
            vsteps = self.validation.steps()
            (vimg, vlabels) = (None, None)
            if not from_training:
                (vimg, vlabels) = (self.validation.images(), self.validation.labels())
            validation = ml_config.ValidationSet(vimg, vlabels, from_training, vsteps)
            self.__training = ml_config.TrainingSpec(batch_size=self._config_dict['batch_size'],
                                                     epochs=self._config_dict['epochs'],
                                                     loss_function=self._config_dict['loss_function'],
                                                     validation=validation,
                                                     steps=self._config_dict['steps'],
                                                     metrics=self._config_dict['metrics'],
                                                     chunk_stride=self._config_dict['chunk_stride'],
                                                     optimizer=self._config_dict['optimizer'],
                                                     experiment_name=self._config_dict['experiment_name'])
        return self.__training


class MLFlowCheckpointsConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self._register_field('frequency', int, 'frequency', None, None,
                             'Frequency in batches to store neural network checkpoints.')
        self._register_field('save_latest', bool, 'save_latest', None, None,
                             'If true, only keep the most recent checkpoint.')

class MLFlowConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self._register_field('enabled', bool, 'enabled', None, None, 'Enable MLFlow.')
        self._register_field('uri', str, None, None, None, 'URI to store MLFlow data.')
        self._register_field('frequency', int, 'frequency', None, validate_positive, 'Frequency to store metrics.')
        self.register_component(MLFlowCheckpointsConfig(), 'checkpoints')

    def uri(self) -> str:
        """
        Returns the URI for MLFlow to store data.
        """
        uri = self._config_dict['uri']
        if uri == 'default':
            uri = 'file://' + os.path.join(appdirs.AppDirs('delta', 'nasa').user_data_dir, 'mlflow')
        return uri

class TensorboardConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self._register_field('enabled', bool, 'enabled', None, None, 'Enable Tensorboard.')
        self._register_field('dir', str, None, None, None, 'Directory to store Tensorboard data.')

    def dir(self) -> str:
        """
        Returns the directory for tensorboard to store to.
        """
        tbd = self._config_dict['dir']
        if tbd == 'default':
            tbd = os.path.join(appdirs.AppDirs('delta', 'nasa').user_data_dir, 'tensorboard')
        return tbd

class DeltaConfig(DeltaConfigComponent):
    """
    DELTA configuration manager.

    Access and control all configuration parameters.
    """
    def load(self, yaml_file: str = None, yaml_str: str = None):
        """
        Loads a config file, then updates the default configuration
        with the loaded values.
        """
        base_path = None
        if yaml_file:
            if not os.path.exists(yaml_file):
                raise Exception('Config file does not exist: ' + yaml_file)
            with open(yaml_file, 'r') as f:
                config_data = yaml.safe_load(f)
            base_path = os.path.normpath(os.path.dirname(yaml_file))
        else:
            config_data = yaml.safe_load(yaml_str)
        self._load_dict(config_data, base_path)

    def setup_arg_parser(self, parser) -> None:
        parser.add_argument('--config', dest='config', action='append', required=False, default=[],
                            help='Load configuration file (can pass multiple times).')
        for c in self._components.values():
            c.setup_arg_parser(parser)

    def parse_args(self, options):
        for c in options.config:
            self.load(c)
        for c in self._components.values():
            c.parse_args(options)
        return options

    def reset(self):
        super().reset()
        self.load(pkg_resources.resource_filename('delta', 'config/delta.yaml'))

    def initialize(self, options, config_files = None):
        """
        Loads the default files unless config_files is specified, in which case it
        loads them. Then loads options (from argparse).
        """
        self.reset()

        if config_files is None:
            dirs = appdirs.AppDirs('delta', 'nasa')
            config_files = [os.path.join(dirs.site_config_dir, 'delta.yaml'),
                            os.path.join(dirs.user_config_dir, 'delta.yaml')]

        for filename in config_files:
            if os.path.exists(filename):
                config.load(filename)

        if options is not None:
            config.parse_args(options)

config = DeltaConfig()

config.register_component(GeneralConfig(), 'general')
config.register_component(DatasetConfig(), 'dataset')
config.register_component(NetworkConfig(), 'network')
config.register_component(TrainingConfig(), 'train')
config.register_component(MLFlowConfig(), 'mlflow')
config.register_component(TensorboardConfig(), 'tensorboard')
