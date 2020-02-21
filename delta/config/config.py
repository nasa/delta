import collections
import os
import os.path

import yaml
import pkg_resources
import appdirs
from delta.imagery import disk_folder_cache
from delta.imagery.sources import image_set
from delta.ml import ml_config

#pylint: disable=W0108

def _recursive_update(d, u, ignore_new):
    """
    Like dict.update, but recursively updates only
    values that have changed in sub-dictionaries.
    """
    for k, v in u.items():
        if not ignore_new and k not in d:
            raise IndexError('Unexpected config value %s.' % (k))
        dv = d.get(k, {})
        if not isinstance(dv, collections.abc.Mapping):
            d[k] = v
        elif isinstance(v, collections.abc.Mapping):
            d[k] = _recursive_update(dv, v, ignore_new)
        else:
            d[k] = v
    return d

__DEFAULT_EXTENSIONS = {'tiff' : '.tiff',
                        'worldview' : '.zip',
                        'tfrecord' : '.tfrecord',
                        'landsat' : '.zip',
                        'npy' : '.npy'}
def __extension(conf):
    if conf['extension'] is None:
        return __DEFAULT_EXTENSIONS.get(conf['type'])
    return conf['extension']

def __find_images(conf, matching_images=None, matching_conf=None):
    '''
    Find the images specified by a given configuration, returning a list of images.
    If matching_images and matching_conf are specified, we find the labels matching these images.
    '''
    images = []
    if (conf['files'] is None) != (conf['file_list'] is None) != (conf['directory'] is None):
        raise  ValueError('Too many image specification methods used.')
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

def _config_to_image_label_sets(images_dict, labels_dict):
    '''
    Takes two configuration subsections and returns (image set, label set)
    '''
    images = __find_images(images_dict)

    if images_dict['directory']:
        if labels_dict['files'] or labels_dict['file_list']:
            raise ValueError('Image directory only supported with label directory.')
        if labels_dict['directory']:
            # remove images in same directory ending with label's extension (can have .tiff and _label.tiff in same dir)
            if os.path.realpath(labels_dict['directory']).startswith(os.path.realpath(images_dict['directory'])):
                label_extension = __extension(labels_dict)
                images = [img for img in images if not img.endswith(label_extension)]

    imageset = image_set.ImageSet(images, images_dict['type'], images_dict['preprocess'], images_dict['nodata_value'])

    if (labels_dict['files'] is None) and (labels_dict['file_list'] is None) and (labels_dict['directory'] is None):
        return (imageset, None)

    labels = __find_images(labels_dict, images, images_dict)

    if len(labels) != len(images):
        raise ValueError('%d images found, but %d labels found.' % (len(images), len(labels)))

    return (imageset, image_set.ImageSet(labels, labels_dict['type'],
                                         labels_dict['preprocess'], labels_dict['nodata_value']))

# images validated when finding the files
def __image_entries(keys, cpre):
    return [
        (keys + ['type'],               None,                str,                None,
         cpre + '-type' if cpre else None, 'Image type (tiff, worldview, landsat, etc.).'),
        (keys + ['files'],               None,               list,               None, None),
        (keys + ['file_list'],           None,               str,                None,
         cpre + '-file-list' if cpre else None, 'Data text file listing images.'),
        (keys + ['directory'],           None,               str,                None,
         cpre + '-dir' if cpre else None, 'Directory to search for images of given extension.'),
        (keys + ['extension'],           None,               str,                None,
         cpre + '-extension' if cpre else None, 'File extension to search for images in given directory.'),
        (keys + ['preprocess'],          None,               bool,               None,            None, None),
        (keys + ['nodata_value'],        None,               float,              None,            None, None)
    ]


# This list contains all the entries expected in the config file, as well as how they are given command line arguments,
# validated and accessed.
#   dictionary entry, method_name, type, validation function, command line, description
_CONFIG_ENTRIES = [
    (['general', 'gpus'],              'gpus',              int,          None,
     'gpus', 'Number of gpus to use.'),
    (['general', 'threads'],           'threads',           int,          lambda x : x is None or x > 0,
     'threads', 'Number of threads to use.'),
    (['general', 'block_size_mb'],     'block_size_mb',     int,          lambda x : x > 0, None,
     'Size of an image block to load in memory at once.'),
    (['general', 'interleave_images'], 'interleave_images', int,          lambda x : x > 0, None,
     'Number of images to interleave at a time when training.'),
    (['general', 'tile_ratio'],        'tile_ratio',        float,        lambda x : x > 0, None,
     'Width to height ratio of blocks to load in images.'),
    (['general', 'cache', 'dir'],      None,                str,          None,             None, None),
    (['general', 'cache', 'limit'],    None,                int,          lambda x : x > 0, None, None),
    (['network', 'chunk_size'],        'chunk_size',        int,          lambda x: x > 0,
     'chunk-size', 'Width of an image chunk to input to the neural network.'),
    (['network', 'output_size'],       'output_size',        int,          lambda x: x > 0,
     'output-size', 'Width of an image chunk to output from the neural network.'),
    (['network', 'classes'],           'classes',           int,          lambda x: x > 0,
     'classes', 'Number of label classes.'),
    (['network', 'model', 'yaml_file'], None,               str,          None,
     'model_description', 'A YAML file describing the network to train.'),
    (['train', 'chunk_stride'],        None,                int,          lambda x: x > 0,
     'chunk-stride', 'Pixels to skip when iterating over chunks. A value of 1 means to take every chunk.'),
    (['train', 'epochs'],              None,                int,          lambda x: x > 0,
     'num-epochs', 'Number of times to repeat training on the dataset.'),
    (['train', 'batch_size'],          None,                int,          lambda x: x > 0,
     'batch-size', 'Features to group into each batch for training.'),
    (['train', 'loss_function'],       None,                str,          None,            None, None),
    (['train', 'metrics'],             None,                list,         None,            None, None),
    (['train', 'steps'],               None,                int,          None,
     'steps', 'Number of steps to train for.'),
    (['train', 'validation', 'steps'], None,                int,          lambda x: x > 0, None, None),
    (['train', 'validation', 'from_training'],        None, bool,         None,            None, None),
    (['train', 'experiment_name'],     None,                str,          None,            None, None),
    (['train', 'optimizer'],           None,                str,          None,            None, None),
    (['mlflow', 'enabled'],   'mlflow_enabled',       bool,               None,            None,
     'Enable MLFlow.'),
    (['mlflow', 'uri'],       'mlflow_uri',           str,                None,            None,
     'URI to store MLFlow data.'),
    (['mlflow', 'frequency'], 'mlflow_freq',          int,                lambda x: x > 0, None,
     'Frequency to store metrics to MLFlow.'),
    (['mlflow', 'checkpoints', 'frequency'], 'mlflow_checkpoint_freq', int,   None,            None,
     'Frequency in batches to store neural network checkpoints in MLFlow.'),
    (['mlflow', 'checkpoints', 'save_latest'], 'mlflow_checkpoint_latest', bool,   None,            None,
     'If true, only store the latest checkpoint.'),
    (['tensorboard', 'enabled'],  'tb_enabled',       bool,               None,            None,
     'Enable tensorboard.'),
    (['tensorboard', 'dir'],      'tb_dir',           str,                None,            None,
     'Directory to store tensorboard data.'),
    (['tensorboard', 'frequency'], 'tb_freq',         int,                lambda x: x > 0, None,
     'Frequency of updates to tensorboard.')
]
_CONFIG_ENTRIES.extend(__image_entries(['images'], 'image'))
_CONFIG_ENTRIES.extend(__image_entries(['labels'], 'label'))
_CONFIG_ENTRIES.extend(__image_entries(['train', 'validation', 'images'], None))
_CONFIG_ENTRIES.extend(__image_entries(['train', 'validation', 'labels'], None))

class DeltaConfig:
    """
    DELTA configuration manager.

    Access and control all configuration parameters.
    """
    def __init__(self):
        """
        Do not create a new instance, only the `delta.config.config`
        singleton should be used.
        """
        self.__config_dict = None
        self._cache_manager = None
        self.__images = None
        self.__labels = None
        self.__training = None
        self._dirs = appdirs.AppDirs('delta', 'nasa')

        self.reset()

    def _get_entry(self, key_list):
        assert len(key_list) >= 1
        a = self.__config_dict
        for k in key_list:
            a = a[k]
        return a

    def export(self) -> str:
        """
        Returns a YAML string of all configuration options.
        """
        return yaml.dump(self.__config_dict)

    def reset(self) -> None:
        """
        Restores the config file to the default state specified in `delta/config/defaults.cfg`.
        """
        self._cache_manager = None
        self.__images = None
        self.__labels = None
        self.__training = None
        self.__config_dict = {}
        self._load(pkg_resources.resource_filename('delta', 'config/delta.yaml'), ignore_new=True)

        # set a few special defaults
        self.__config_dict['general']['cache']['dir'] = self._dirs.user_cache_dir
        self.__config_dict['mlflow']['uri'] = 'file://' + \
                       os.path.join(self._dirs.user_data_dir, 'mlflow')
        self.__config_dict['tensorboard']['dir'] = \
                os.path.join(self._dirs.user_data_dir, 'tensorboard')

    def load(self, yaml_file: str = None, yaml_str: str = None):
        """
        Loads a config file, then updates the default configuration
        with the loaded values.
        """
        self._load(yaml_file, yaml_str)

    def _load(self, yaml_file=None, yaml_str=None, ignore_new=False):
        if yaml_file:
            if not os.path.exists(yaml_file):
                raise Exception('Config file does not exist: ' + yaml_file)
            with open(yaml_file, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = yaml.safe_load(yaml_str)
        def normalize_path(path):
            path = os.path.expanduser(path)
            # make relative paths relative to this config file
            if yaml_file:
                path = os.path.normpath(os.path.join(os.path.dirname(yaml_file), path))
            return path
        # expand paths to use relative ones to this config file
        def recursive_normalize(d):
            for k, v in d.items():
                if isinstance(v, collections.abc.Mapping):
                    recursive_normalize(v)
                else:
                    if k == 'yaml_file' and pkg_resources.resource_exists('delta', os.path.join('config', v)):
                        continue
                    if ('dir' in k or 'file' in k):
                        if isinstance(v, str):
                            d[k] = normalize_path(v)
                        elif isinstance(v, list):
                            d[k] = [normalize_path(item) for item in v]
        recursive_normalize(config_data)

        self.__config_dict = _recursive_update(self.__config_dict, config_data, ignore_new)

        # overwrite model entirely if updated (don't want combined layers from multiple files)
        if 'network' in config_data and 'model' in config_data['network']:
            m = config_data['network']['model']
            for k in ['yaml_file', 'layers', 'params']:
                if not k in m:
                    m[k] = None
            self.__config_dict['network']['model'] = m

        self._validate()

    def _validate(self):
        for e in _CONFIG_ENTRIES:
            v = self._get_entry(e[0])
            if v is not None and not isinstance(v, e[2]):
                raise TypeError('%s must be of type %s, is %s.' % (e[0][-1], e[2], v))
            if e[3] and not e[3](v):
                raise ValueError('Value %s for %s is invalid.' % (v, e[0][-1]))

    def cache_manager(self) -> disk_folder_cache.DiskCache:
        if self._cache_manager is None:
            self._cache_manager = disk_folder_cache.DiskCache(self.__config_dict['general']['cache']['dir'],
                                                              self.__config_dict['general']['cache']['limit'])
        return self._cache_manager

    def __load_images_labels(self, image_keys, label_keys):
        images = self._get_entry(image_keys)
        labels = self._get_entry(label_keys)
        return _config_to_image_label_sets(images, labels)

    def images(self) -> image_set.ImageSet:
        """
        Returns the training images.
        """
        if self.__images is None:
            (self.__images, self.__labels) = self.__load_images_labels(['images'], ['labels'])
        return self.__images

    def labels(self) -> image_set.ImageSet:
        """
        Returns the label images.
        """
        if self.__labels is None:
            (self.__images, self.__labels) = self.__load_images_labels(['images'], ['labels'])
        return self.__labels

    def model_dict(self) -> dict:
        """
        Returns a dictionary representing the network model for use by `delta.ml.model_parser`.
        """
        model = self._get_entry(['network', 'model'])
        yaml_file = model['yaml_file']
        if yaml_file is not None:
            if model['layers'] is not None:
                raise ValueError('Specified both yaml file and layers in model.')

            resource = os.path.join('config', yaml_file)
            if not os.path.exists(yaml_file) and pkg_resources.resource_exists('delta', resource):
                yaml_file = pkg_resources.resource_filename('delta', resource)
            if not os.path.exists(yaml_file):
                raise ValueError('Model yaml_file does not exist: ' + yaml_file)
            with open(yaml_file, 'r') as f:
                return yaml.safe_load(f)
        return model

    def training(self) -> ml_config.TrainingSpec:
        """
        Returns the options configuring training.
        """
        if self.__training is not None:
            return self.__training
        from_training = self._get_entry(['train', 'validation', 'from_training'])
        vsteps = self._get_entry(['train', 'validation', 'steps'])
        (vimg, vlabels) = (None, None)
        if not from_training:
            (vimg, vlabels) = self.__load_images_labels(['train', 'validation', 'images'],
                                                        ['train', 'validation', 'labels'])
        validation = ml_config.ValidationSet(vimg, vlabels, from_training, vsteps)
        self.__training = ml_config.TrainingSpec(batch_size=self._get_entry(['train', 'batch_size']),
                                                 epochs=self._get_entry(['train', 'epochs']),
                                                 loss_function=self._get_entry(['train', 'loss_function']),
                                                 validation=validation,
                                                 steps=self._get_entry(['train', 'steps']),
                                                 metrics=self._get_entry(['train', 'metrics']),
                                                 chunk_stride=self._get_entry(['train', 'chunk_stride']),
                                                 optimizer=self._get_entry(['train', 'optimizer']),
                                                 experiment_name=self._get_entry(['train', 'experiment_name']))
        return self.__training

    def __add_arg_group(self, group, group_key):#pylint:disable=no-self-use
        '''Add command line arguments for the given group.'''
        for e in _CONFIG_ENTRIES:
            if e[0][0] == group_key and e[4] is not None:
                group.add_argument('--' + e[4], dest=e[4].replace('-', '_'), required=False, type=e[2], help=e[5])

    def setup_arg_parser(self, parser, general=True, images=True, labels=True, train=False) -> None:
        """
        Setup the ArgParser parser to allow the specified options.

         * **general**: General options which don't fit in another group.
         * **images**: Specify input images.
         * **labels**: Specify labels corresponding to the images.
         * **train**: Specify options for training a neural network.
        """
        group = parser.add_argument_group('General')
        group.add_argument('--config', dest='config', action='append', required=False, default=[],
                           help='Load configuration file (can pass multiple times).')
        if general:
            self.__add_arg_group(group, 'general')

        if images:
            group = parser.add_argument_group('Input Data')
            self.__add_arg_group(group, 'images')
            group.add_argument("--image", dest="image", required=False,
                               help="Specify a single image file.")
        if labels:
            self.__add_arg_group(group, 'labels')
            group.add_argument("--label", dest="label", required=False,
                               help="Specify a single label file.")

        if train:
            group = parser.add_argument_group('Machine Learning')
            self.__add_arg_group(group, 'network')
            self.__add_arg_group(group, 'train')

    def parse_args(self, options):
        """
        Parse an options extracted from an ArgParser configured with
        `setup_arg_parser` and override the appropriate
        configuration values.
        """
        for c in options.config:
            self.load(c)

        c = self.__config_dict
        if hasattr(options, 'image') and options.image:
            c['images']['files'] = [options.image]
        if hasattr(options, 'label') and options.label:
            c['labels']['files'] = [options.label]
        # load all the command line arguments into the config_dict
        for e in _CONFIG_ENTRIES:
            if e[4] is None:
                continue
            name = e[4].replace('-', '_')
            if not hasattr(options, name):
                continue
            v = getattr(options, name)
            if v is None:
                continue
            a = self.__config_dict
            for k in e[0][:-1]:
                a = a[k]
            a[e[0][-1]] = v

        self._validate()
        return options

# make accessor functions for DeltaConfig based on list
def _create_accessor(key_list, name, doc, return_type):
    def accessor(self) -> return_type:
        return self._get_entry(key_list)#pylint:disable=protected-access
    accessor.__name__ = name
    accessor.__doc__ = doc
    setattr(DeltaConfig, name, accessor)

def __initialize_delta_config():
    for e in _CONFIG_ENTRIES:
        if e[1] is None:
            continue
        _create_accessor(e[0], e[1], e[5], e[2])

__initialize_delta_config()
config = DeltaConfig()

def __load_initial_config():
    # only contains things not in default config file
    global config #pylint: disable=global-statement
    dirs = appdirs.AppDirs('delta', 'nasa')
    DEFAULT_CONFIG_FILES = [os.path.join(dirs.site_config_dir, 'delta.yaml'),
                            os.path.join(dirs.user_config_dir, 'delta.yaml')]

    for filename in DEFAULT_CONFIG_FILES:
        if os.path.exists(filename):
            config.load(filename)

__load_initial_config()
