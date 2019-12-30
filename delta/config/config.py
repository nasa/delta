import collections
import os
import os.path

import yaml
import pkg_resources
import appdirs
from delta.imagery import disk_folder_cache

def recursive_update(d, u, ignore_new):
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
            d[k] = recursive_update(dv, v, ignore_new)
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

class ImageSet:
    def __init__(self, images, image_type, preprocess=False):
        self._images = images
        self._image_type = image_type
        self._preprocess = preprocess

    def type(self):
        return self._image_type
    def preprocess(self):
        return self._preprocess
    def __len__(self):
        return len(self._images)
    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError('Index %s out of range.' % (index))
        return self._images[index]
    def __iter__(self):
        return self._images.__iter__()

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

    image_set = ImageSet(images, images_dict['type'], images_dict['preprocess'])

    if (labels_dict['files'] is None) and (labels_dict['file_list'] is None) and (labels_dict['directory'] is None):
        return (image_set, None)

    labels = __find_images(labels_dict, images, images_dict)

    if len(labels) != len(images):
        raise ValueError('%d images found, but %d labels found.' % (len(images), len(labels)))

    return (image_set, ImageSet(labels, labels_dict['type'], labels_dict['preprocess']))

# This list contains all the entries expected in the config file, as well as how they are given command line arguments,
# validated and accessed.
#   dictionary entry, method_name, type, validation function, command line, description
_CONFIG_ENTRIES = [
        (['general', 'gpus'],              'gpus',              int,                None,
         'gpus', 'Number of gpus to use.'),
        (['general', 'threads'],           'threads',           int,                lambda x : x is None or x > 0,
         'threads', 'Number of threads to use.'),
        (['general', 'block_size_mb'],     'block_size_mb',     int,                lambda x : x > 0, None, None),
        (['general', 'interleave_images'], 'interleave_images', int,                lambda x : x > 0, None, None),
        (['general', 'tile_ratio'],        'tile_ratio',        float,              lambda x : x > 0, None, None),
        (['general', 'cache', 'dir'],      None,                str,                None,             None, None),
        (['general', 'cache', 'limit'],    None,                int,                lambda x : x > 0, None, None),
        # images and labels validated when finding the files
        (['images', 'type'],               None,                str,                None,
         'image-type', 'Image type (tiff, worldview, landsat, etc.).'),
        (['images', 'files'],               None,               list,               None, None),
        (['images', 'file_list'],           None,               str,                None,
         'image-file-list', 'Data text file listing images.'),
        (['images', 'directory'],           None,               str,                None,
         'image-dir', 'Directory to search for images of given extension.'),
        (['images', 'extension'],           None,               str,                None,
         'image-extension', 'File extension to search for images in given directory.'),
        (['images', 'preprocess'],          None,               bool,               None,            None, None),
        (['labels', 'type'],                None,               str,                None,
         'label-type', 'Label type (tiff, worldview, landsat, etc.).'),
        (['labels', 'files'],               None,               list,               None, None),
        (['labels', 'file_list'],           None,               str,                None,
         'label-file-list', 'Data text file listing images.'),
        (['labels', 'directory'],           None,               str,                None,
         'labels-dir', 'Directory to search for images of given extension.'),
        (['labels', 'extension'],           None,               str,                None,
         'label-extension', 'File extension to search for images in given directory.'),
        (['labels', 'preprocess'],          None,               bool,               None,            None, None),
        (['ml', 'chunk_size'],              'chunk_size',       int,                lambda x: x > 0 and x % 2 == 1,
         'chunk-size', 'Width of an image chunk to process at once.'),
        (['ml', 'chunk_stride'],            'chunk_stride',     int,                lambda x: x > 0,
         'chunk-stride', 'Pixels to skip when iterating over chunks. A value of 1 means to take every chunk.'),
        (['ml', 'num_epochs'],              'num_epochs',       int,                lambda x: x > 0,
         'num-epochs', 'Number of times to repeat training on the dataset.'),
        (['ml', 'batch_size'],              'batch_size',       int,                lambda x: x > 0,
         'batch-size', 'Features to group into each batch for training.'),
        (['ml', 'num_classes'],             'num_classes',      int,                lambda x: x > 0,
         'num-classes', 'Number of label classes.'),
        (['ml', 'loss_function'],           'loss_function',    str,                None,
         'loss-fn', 'Tensorflow loss function to use.'),
        (['ml', 'mlflow', 'enabled'],       'mlflow_enabled',   bool,               None,            None, None),
        (['ml', 'mlflow', 'uri'],           'mlflow_uri',       str,                None,            None, None),
        (['ml', 'tensorboard', 'enabled'],  'tb_enabled',       bool,               None,            None, None),
        (['ml', 'tensorboard', 'dir'],      'tb_dir',           str,                None,            None, None),
        (['ml', 'checkpoint_dir'],          'checkpoint_dir',   str,                None,            None, None)
]

class DeltaConfig:
    def __init__(self):
        self.__config_dict = None
        self._cache_manager = None
        self.__images = None
        self.__labels = None

        self.reset()

    def _get_entry(self, key_list):
        assert len(key_list) >= 1
        a = self.__config_dict
        for k in key_list:
            a = a[k]
        return a

    def reset(self):
        """
        Restores the config file to the default state specified in defaults.cfg.
        """
        self._cache_manager = None
        self.__images = None
        self.__labels = None
        self.__config_dict = {}
        self.load(pkg_resources.resource_filename('delta', 'config/delta.yaml'), ignore_new=True)

        # set a few special defaults
        self.__config_dict['general']['cache']['dir'] = appdirs.AppDirs('delta', 'nasa').user_cache_dir
        self.__config_dict['ml']['mlflow_uri'] = 'file://' + \
                       os.path.join(appdirs.AppDirs('delta', 'nasa').user_data_dir, 'mlflow')
        self.__config_dict['ml']['tb_dir'] = os.path.join(appdirs.AppDirs('delta', 'nasa').user_data_dir, 'tensorboard')

    def load(self, yaml_file=None, yaml_str=None, ignore_new=False):
        """
        Loads a config file, then updates the default configuration
        with the loaded values. Can be passed as either a file or a string.
        """
        if yaml_file:
            if not os.path.exists(yaml_file):
                raise Exception('Config file does not exist: ' + yaml_file)
            with open(yaml_file, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = yaml.safe_load(yaml_str)
        # expand paths to use relative ones to this config file
        def recursive_normalize(d):
            for k, v in d.items():
                if isinstance(v, collections.abc.Mapping):
                    recursive_normalize(v)
                else:
                    if ('dir' in k or 'file' in k) and isinstance(v, str):
                        v = os.path.expanduser(v)
                        # make relative paths relative to this config file
                        if yaml_file:
                            d[k] = os.path.normpath(os.path.join(os.path.dirname(yaml_file), v))
        recursive_normalize(config_data)

        self.__config_dict = recursive_update(self.__config_dict, config_data, ignore_new)
        self._validate()

    def _validate(self):
        for e in _CONFIG_ENTRIES:
            v = self._get_entry(e[0])
            if v is not None and not isinstance(v, e[2]):
                raise TypeError('%s must be of type %s, is %s.' % (e[0][-1], e[2], v))
            if e[3] and not e[3](v):
                raise ValueError('Value %s for %s is invalid.' % (v, e[0][-1]))

    def set_value(self, group, name, value):
        self.__config_dict[group][name] = value

    def cache_manager(self):
        if self._cache_manager is None:
            self._cache_manager = disk_folder_cache.DiskCache(self.__config_dict['general']['cache']['dir'],
                                                              self.__config_dict['general']['cache']['limit'])
        return self._cache_manager

    def __load_images_labels(self):
        (self.__images, self.__labels) = _config_to_image_label_sets(self.__config_dict['images'],
                                                                     self.__config_dict['labels'])

    def images(self):
        if self.__images is None:
            self.__load_images_labels()
        return self.__images

    def labels(self):
        if self.__labels is None:
            self.__load_images_labels()
        return self.__labels

    def __add_arg_group(self, group, group_key):#pylint:disable=no-self-use
        '''Add command line arguments for the given group.'''
        for e in _CONFIG_ENTRIES:
            if e[0][0] == group_key and e[4] is not None:
                group.add_argument('--' + e[4], dest=e[4].replace('-', '_'), required=False, type=e[2], help=e[5])

    def setup_arg_parser(self, parser, general=True, images=True, labels=True, ml=True):
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

        if ml:
            group = parser.add_argument_group('Machine Learning')
            self.__add_arg_group(group, 'ml')

    def parse_args(self, options):
        for c in options.config:
            self.load(c)

        c = self.__config_dict
        if hasattr(options, 'image') and options.image:
            c['images']['files'] = [options.image]
        if hasattr(options, 'lable') and options.label:
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
def _create_accessor(key_list, name, doc):
    def accessor(self):
        return self._get_entry(key_list)#pylint:disable=protected-access
    accessor.__name__ = name
    accessor.__doc__ = doc
    setattr(DeltaConfig, name, accessor)

def __initialize_delta_config():
    for e in _CONFIG_ENTRIES:
        if e[1] is None:
            continue
        _create_accessor(e[0], e[1], e[5])

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
