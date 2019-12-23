import argparse
import collections
import yaml
import os
import os.path
import sys

import numpy as np
import pkg_resources
import appdirs
from delta.imagery import disk_folder_cache

def recursive_update(d, u):
    """
    Like dict.update, but recursively updates only
    values that have changed in sub-dictionaries.
    """
    for k, v in u.items():
        dv = d.get(k, {})
        if not isinstance(dv, collections.abc.Mapping):
            d[k] = v
        elif isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(dv, v)
        else:
            d[k] = v
    return d

class DatasetConfig:#pylint:disable=too-many-instance-attributes
    def __init__(self, images_dict, labels_dict):
        DEFAULT_EXTENSIONS = {'tiff' : '.tiff',
                              'worldview' : '.zip',
                              'tfrecord' : '.tfrecord',
                              'landsat' : '.zip',
                              'npy' : '.npy'
                             }

        self._image_type = images_dict['type']
        self._image_directory = images_dict['directory']
        self._image_extension = images_dict['extension']
        self._image_file_list = images_dict['file_list']
        self._image_files = images_dict['files']

        self._label_type = labels_dict['type']
        self._label_directory = labels_dict['directory']
        self._label_extension = labels_dict['extension']
        self._label_file_list = labels_dict['file_list']
        self._label_files = labels_dict['files']

        if self._image_directory and self._image_extension is None:
            if self._image_type in DEFAULT_EXTENSIONS:
                self._image_extension = DEFAULT_EXTENSIONS[self._image_type]
        if self._label_directory and self._label_extension is None:
            if self._label_type in DEFAULT_EXTENSIONS:
                self._label_extension = '_label' + DEFAULT_EXTENSIONS[self._label_type]

        self._label = config_dict['label']
        self._label_directory = config_dict['label_directory']
        self._label_extension = config_dict['label_extension']

        self._num_threads = config_dict['num_input_threads']
        self._max_block_size = config_dict['max_block_size']
        self._num_interleave_images = config_dict['num_interleave_images']
        self._tile_ratio = config_dict['tile_ratio']
        self._preprocess = config_dict['preprocess']

        (self._images, self._labels) = self.__find_images()

    def data_directory(self):
        return self._data_directory
    def label_directory(self):
        return self._label_directory
    def image_type(self):
        return self._image_type
    def label_type(self):
        return self._label_type
    def preprocess(self):
        return self._preprocess

    def _get_label(self, image_path):
        """Returns the path to the expected label for for the given input image file"""

        # Label file should have the same name but different extension in the label folder
        rel_path   = os.path.relpath(image_path, self._data_directory)
        label_path = os.path.join(self._label_directory, rel_path)
        label_path = os.path.splitext(label_path)[0] + self._label_extension
        if not os.path.exists(label_path):
            raise Exception('Error: Expected label file to exist at path: ' + label_path)
        return label_path

    def __find_images(self):
        """List all of the files specified by the configuration.
           If a label folder is provided, look for corresponding label files which
           have the same relative path in that folder but ending with "_label.tif".
        """

        if self._label_directory:
            if not os.path.exists(self._label_directory):
                raise Exception('Supplied label folder does not exist: ' + self._label_directory)

        image_files = []
        if self._label_directory or self._label:
            label_files = []
        else:
            label_files = None

        if self._image:
            image_files.append(self._image)
            if self._label:
                label_files.append(self._label)
            elif self._label_directory:
                label_files.append(self._get_label(self._image))
        elif self._data_directory:
            for root, dummy_directories, filenames in os.walk(self._data_directory):
                for filename in filenames:
                    if filename.endswith(self._image_extension):
                        if self._label_extension and filename.endswith(self._label_extension):
                            continue
                        path = os.path.join(root, filename.strip())
                        image_files.append(path)

                        if self._label_directory:
                            label_files.append(self._get_label(path))
            if not image_files:
                print('No images ending in %s found in %s.' % (self._image_extension, self._data_directory))
        elif self._data_file_list:
            with open(self._data_file_list, 'r') as f:
                for line in f:
                    parts = line.split()
                    image_files.append(parts[0])
                    if label_files and len(parts) >= 2:
                        label_files.append(parts[1])

        image_files = np.array(image_files)
        if label_files:
            label_files = np.array(label_files)
        return (image_files, label_files)

    def images(self):
        """Returns (image_files, label_files)."""
        return (self._images, self._labels)

    def num_images(self):
        return len(self._images)

    def num_labels(self):
        return len(self._labels)

    def image(self, index):
        if index < 0 or index >= self.num_images():
            raise IndexError('Index %s out of range.' % (index))
        return self._images[index]

    def label(self, index):
        if index < 0 or index >= len(self._labels):
            raise IndexError('Index %s out of range.' % (index))
        return self._labels[index]

class DeltaConfig:
    def __init__(self):
        self.__config_dict = None
        self._cache_manager = None
        self.reset()

    def reset(self):
        """
        Restores the config file to the default state specified in defaults.cfg.
        """
        self.__config_dict = {}
        self.__config_dict['general'] = {}
        self.__config_dict['general']['cache'] = {}
        self.__config_dict['general']['cache']['cache_dir'] = appdirs.AppDirs('delta', 'nasa').user_cache_dir
        self.load(pkg_resources.resource_filename('delta', 'config/delta.yaml'), ignore_new=True)
        self._cache_manager = None

    def load(self, config_path, ignore_new=False):
        """
        Loads a config file, then updates the default configuration
        with the loaded values.
        """
        if not os.path.exists(config_path):
            raise Exception('Config file does not exist: ' + config_path)

        config_data = yaml.safe_load(config_path)
        # expand paths to use relative ones to this config file
        def recurse_normalize(d):
            for k, v in d.items():
                if isinstance(v, collections.abc.Mapping):
                    __recurse_normalize(v)
                else:
                    if ('dir' in k or 'file' in k) and isinstance(v, str):
                        v = os.path.expanduser(v)
                        # make relative paths relative to this config file
                        d[k] = os.path.normpath(os.path.join(os.path.dirname(config_path), v))
        recursive_normalize(config_data)

        self.__config_dict = recursive_update(self.__config_dict, config_data)
        self.__validate()

    def __validate(self):
        if self.__config_dict['ml']['chunk_size'] % 2 == 0:
            raise ValueError('chunk_size must be odd.')
        if self.__config_dict['images']['type'] is None:
            raise ValueError('Must specify a valid image type.')

    def set_value(self, group, name, value):
        self.__config_dict[group][name] = value

    def num_gpus(self):
        return self.__config_dict['general']['num_gpus']
    def num_threads(self):
        return self.__config_dict['general']['threads']
    def block_size_mb(self):
        return self.__config_dict['general']['block_size_mb']
    def interleave_images(self):
        return self.__config_dict['general']['interleave_images']
    def tile_ratio(self):
        return self.__config_dict['general']['tile_ratio']

    def dataset(self):
        return DatasetConfig(self.__config_dict['images'], self.__config_dict['labels'])

    def chunk_size(self):
        return self.__config_dict['ml']['chunk_size']
    def chunk_stride(self):
        return self.__config_dict['ml']['chunk_stride']
    def batch_size(self):
        return self.__config_dict['ml']['batch_size']
    def num_epochs(self):
        return self.__config_dict['ml']['num_epochs']
    def output_dir(self):
        return self.__config_dict['ml']['output_dir']
    def model_dir(self):
        return self.__config_dict['ml']['model_dir']
    def model_dest_name(self):
        return self.__config_dict['ml']['model_dest_name']
    def num_classes(self):
        return self.__config_dict['ml']['num_classes']
    def loss_function(self):
        return self.__config_dict['ml']['loss_fn']

    def cache_manager(self):
        if self._cache_manager is None:
            self._cache_manager = disk_folder_cache.DiskCache(self.__config_dict['general']['cache']['cache_dir'],
                                                              self.__config_dict['general']['cache']['cache_limit'])
        return self._cache_manager

    def parse_args(self, parser, args, labels=True, ml=True):#pylint:disable=too-many-branches
        group = parser.add_argument_group('General')
        group.add_argument("--num-gpus", dest="num_gpus", required=False, type=int,
                           help="Try to use this many GPUs.")

        group = parser.add_argument_group('Input Data')

        group.add_argument('--data-config', dest='data_config', required=False,
                           help='Dataset configuration file.')

        group.add_argument("--image", dest="image", required=False,
                           help="Specify a single image file.")
        group.add_argument("--image-dir", dest="image_dir", required=False,
                           help="Specify data folder to search for images.")
        group.add_argument("--image-file-list", dest="image_file_list", required=False,
                           help="Specify data text file listing images.")
        group.add_argument("--image-type", dest="image_type", required=False,
                           help="Image type (tiff, worldview, landsat, etc.).")
        group.add_argument("--image-extension", dest="image_extension", required=False,
                           help="File type for images (.tfrecord, .tiff, .tar.gz, etc.).")
        if labels:
            group.add_argument("--label", dest="label", required=False,
                               help="Specify a single label file.")
            group.add_argument("--label-dir", dest="label_dir", required=False,
                               help="Specify label folder instead of supplying config file.")
            group.add_argument("--label-type", dest="label_type", required=False,
                               help="Label file type.")
            group.add_argument("--label-extension", dest="label_extension", required=False,
                               help="File extension for labels (.tfrecord, .tiff, etc.).")

        if ml:
            group = parser.add_argument_group('Machine Learning')
            group.add_argument('--ml-config', dest='ml_config', required=False,
                               help='ML configuration file.')
            group.add_argument("--chunk-size", dest="chunk_size", required=False, type=int,
                               help="Width of an image chunk to process at once.")
            group.add_argument("--chunk-stride", dest="chunk_stride", required=False, type=int,
                               help="Pixels to skip between chunks. A value of 1 means every chunk.")
            group.add_argument("--batch-size", dest="batch_size", required=False, type=int,
                               help="Number of features in a batch.")
            group.add_argument("--num-epochs", dest="num_epochs", required=False, type=int,
                               help="Number of times to run through all the features.")
            group.add_argument("--output-dir", dest="output_dir", required=False,
                               help="Folder to store all output in.")

        try:
            options = parser.parse_args(args[1:])
        except argparse.ArgumentError:
            parser.print_help(sys.stderr)
            sys.exit(1)

        if options.data_config:
            config.load(options.data_config)
        if ml:
            if options.ml_config:
                config.load(options.ml_config)

        c = self.__config_dict
        if options.image:
            c['images']['files'] = [options.image]
        if options.image_dir:
            c['images']['directory'] = options.image_dir
        if options.image_file_list:
            c['images']['file_list'] = options.image_file_list
        if options.image_type:
            c['images']['type'] = options.image_type
        if options.image_extension:
            c['images']['extension'] = options.image_extension
        if labels:
            if options.label:
                c['labels']['files'] = [options.label]
            if options.label_dir:
                c['labels']['directory'] = options.label_dir
            if options.label_type:
                c['labels']['type'] = options.label_type
            if options.label_extension:
                c['labels']['extension'] = options.label_extension

        if ml:
            if options.batch_size:
                c['ml']['batch_size'] = options.batch_size
            if options.chunk_size:
                c['ml']['chunk_size'] = options.chunk_size
            if options.chunk_stride:
                c['ml']['chunk_stride'] = options.chunk_stride
            if options.num_epochs:
                c['ml']['num_epochs'] = options.num_epochs
            if options.output_dir:
                c['ml']['output_dir'] = options.output_dir

        self.__validate()
        return options

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
