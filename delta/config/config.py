import argparse
import collections
import configparser
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
        if not isinstance(dv, collections.Mapping):
            d[k] = v
        elif isinstance(v, collections.Mapping):
            d[k] = recursive_update(dv, v)
        else:
            d[k] = v
    return d

class DatasetConfig:
    def __init__(self, config_dict):
        DEFAULT_EXTENSIONS = {'tiff' : '.tiff',
                              'worldview' : '.zip',
                              'tfrecord' : '.tfrecord',
                              'landsat' : '.zip'
                             }

        self._image_type = config_dict['image_type']
        self._file_type = config_dict['file_type']
        self._label_type = config_dict['label_type']

        self._data_directory = config_dict['data_directory']
        self._image_extension = config_dict['extension']
        self._data_file_list = config_dict['data_file_list']
        if self._data_directory and self._image_extension is None:
            if self._file_type in DEFAULT_EXTENSIONS:
                self._image_extension = DEFAULT_EXTENSIONS[self._file_type]

        self._label_directory = config_dict['label_directory']
        self._label_extension = config_dict['label_extension']
        if self._label_directory and self._label_extension is None:
            if self._label_type in DEFAULT_EXTENSIONS:
                self._label_extension = '_label' + DEFAULT_EXTENSIONS[self._label_type]

        self._num_threads = config_dict['num_input_threads']
        self._shuffle_buffer_size = config_dict['shuffle_buffer_size']
        self._max_block_size = config_dict['max_block_size']
        self._num_interleave_images = config_dict['num_interleave_images']
        self._tile_ratio = config_dict['tile_ratio']

    def data_directory(self):
        return self._data_directory
    def label_directory(self):
        return self._label_directory
    def image_type(self):
        return self._image_type
    def file_type(self):
        return self._file_type
    def label_type(self):
        return self._label_type
    def num_threads(self):
        return self._num_threads
    def shuffle_buffer_size(self):
        return self._shuffle_buffer_size
    def max_block_size(self):
        return self._max_block_size
    def num_interleave_images(self):
        return self._num_interleave_images
    def tile_ratio(self):
        return self._tile_ratio

    def _get_label(self, image_path):
        """Returns the path to the expected label for for the given input image file"""

        # Label file should have the same name but different extension in the label folder
        rel_path   = os.path.relpath(image_path, self._data_directory)
        label_path = os.path.join(self._label_directory, rel_path)
        label_path = os.path.splitext(label_path)[0] + self._label_extension
        if not os.path.exists(label_path):
            raise Exception('Error: Expected label file to exist at path: ' + label_path)
        return label_path

    def images(self):
        """List all of the files specified by the configuration.
           If a label folder is provided, look for corresponding label files which
           have the same relative path in that folder but ending with "_label.tif".
           Returns (image_files, label_files)
        """

        if self._label_directory:
            if not os.path.exists(self._label_directory):
                raise Exception('Supplied label folder does not exist: ' + self._label_directory)

        image_files = []
        if self._label_directory:
            label_files = []
        else:
            label_files = None

        if self._data_directory:
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
        indices = np.arange(len(image_files))
        np.random.shuffle(indices)

        shuffle_images = image_files[indices]
        if label_files:
            label_files = np.array(label_files)
            shuffle_labels = label_files[indices]
        else:
            shuffle_labels = None
        return (shuffle_images, shuffle_labels)

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
        self.__config_dict['cache'] = {}
        self.__config_dict['cache']['cache_dir'] = appdirs.AppDirs('delta', 'nasa').user_cache_dir
        self.load(pkg_resources.resource_filename('delta', 'config/delta.cfg'), ignore_new=True)
        self._cache_manager = None

    def load(self, config_path, ignore_new=False):
        """
        Loads a config file, then updates the default configuration
        with the loaded values.
        """
        if not os.path.exists(config_path):
            raise Exception('Config file does not exist: ' + config_path)
        config_reader = configparser.ConfigParser()

        try:
            config_reader.read(config_path)
        except IndexError:
            raise Exception('Failed to read config file: ' + config_path)

        # Convert to a dictionary
        config_data = {s:dict(config_reader.items(s)) for s in config_reader.sections()}

        # Make sure all sections are there
        for section, items in config_data.items():
            for name, value in items.items():
                if not ignore_new and (section not in self.__config_dict or name not in self.__config_dict[section]):
                    print('Unrecognized key %s:%s in config file %s.' % (section, name, config_path), file=sys.stderr)
                    sys.exit(1)
                value = os.path.expandvars(value)
                if value.lower() == 'none' or value == '': # Useful in some cases
                    value = None
                elif 'folder' in name or 'directory' in name:
                    value = os.path.expanduser(value)
                    # make relative paths relative to this config file
                    value = os.path.normpath(os.path.join(os.path.dirname(config_path), value))
                else:
                    try: # Convert eligible values to integers
                        value = int(value)
                    except (ValueError, TypeError):
                        pass
                config_data[section][name] = value

        self.__config_dict = recursive_update(self.__config_dict, config_data)

    def set_value(self, group, name, value):
        self.__config_dict[group][name] = value

    def num_gpus(self):
        return self.__config_dict['general']['num_gpus']

    def dataset(self):
        return DatasetConfig(self.__config_dict['input_dataset'])

    def chunk_size(self):
        return self.__config_dict['ml']['chunk_size']
    def chunk_stride(self):
        return self.__config_dict['ml']['chunk_stride']
    def batch_size(self):
        return self.__config_dict['ml']['batch_size']
    def num_epochs(self):
        return self.__config_dict['ml']['num_epochs']
    def output_folder(self):
        return self.__config_dict['ml']['output_folder']
    def model_folder(self):
        return self.__config_dict['ml']['model_folder']
    def model_dest_name(self):
        return self.__config_dict['ml']['model_dest_name']
    def num_hidden(self):
        return self.__config_dict['ml']['num_hidden']
    def num_classes(self):
        return self.__config_dict['ml']['num_classes']
    def loss_function(self):
        return self.__config_dict['ml']['loss_fn']

    def cache_manager(self):
        if self._cache_manager is None:
            self._cache_manager = disk_folder_cache.DiskCache(self.__config_dict['cache']['cache_dir'],
                                                              self.__config_dict['cache']['cache_limit'])
        return self._cache_manager

    def parse_args(self, parser, args, labels=True, ml=True):#pylint:disable=too-many-branches
        group = parser.add_argument_group('General')
        group.add_argument("--num-gpus", dest="num_gpus", required=False, type=int,
                           help="Try to use this many GPUs.")

        group = parser.add_argument_group('Input Data')

        group.add_argument('--data-config', dest='data_config', required=False,
                           help='Dataset configuration file.')

        group.add_argument("--data-folder", dest="data_folder", required=False,
                           help="Specify data folder instead of supplying config file.")
        group.add_argument("--data-file-list", dest="data_file_list", required=False,
                           help="Specify data folder instead of supplying config file.")
        group.add_argument("--image-type", dest="image_type", required=False,
                           help="Image type (tiff, worldview, landsat, etc.).")
        group.add_argument("--file-type", dest="file_type", required=False,
                           help="File type (tfrecord, tiff, worldview, landsat, etc.).")
        group.add_argument("--image-extension", dest="image_extension", required=False,
                           help="File type for images (.tfrecord, .tiff, .tar.gz, etc.).")
        if labels:
            group.add_argument("--label-folder", dest="label_folder", required=False,
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
            group.add_argument("--output-folder", dest="output_folder", required=False,
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
        if options.label_folder:
            c['input_dataset']['label_directory'] = options.label_folder
        if options.data_folder:
            c['input_dataset']['data_directory'] = options.data_folder
        if options.data_file_list:
            c['input_dataset']['data_file_list'] = options.data_file_list
        if c['input_dataset']['data_directory'] is None != c['input_dataset']['data_file_list'] is None:
            print('Must specify one of data_directory or data_file_list.', file=sys.stderr)
            sys.exit(0)

        if options.image_type:
            c['input_dataset']['image_type'] = options.image_type
        if c['input_dataset']['image_type'] is None:
            print('Must specify an image_type.', file=sys.stderr)
            sys.exit(0)
        if options.file_type:
            c['input_dataset']['file_type'] = options.file_type
        if options.image_extension:
            c['input_dataset']['extension'] = options.image_extension
        if options.label_type:
            c['input_dataset']['label_type'] = options.label_type
        if options.label_extension:
            c['input_dataset']['label_extension'] = options.label_extension

        if ml:
            if options.batch_size:
                c['ml']['batch_size'] = options.batch_size
            if options.chunk_size:
                c['ml']['chunk_size'] = options.chunk_size
            if options.chunk_stride:
                c['ml']['chunk_stride'] = options.chunk_stride
            if options.num_epochs:
                c['ml']['num_epochs'] = options.num_epochs
            if options.output_folder:
                c['ml']['output_folder'] = options.output_folder

        return options

config = DeltaConfig()

def __load_initial_config():
    # only contains things not in default config file
    global config #pylint: disable=global-statement
    dirs = appdirs.AppDirs('delta', 'nasa')
    DEFAULT_CONFIG_FILES = [os.path.join(dirs.site_config_dir, 'delta.cfg'),
                            os.path.join(dirs.user_config_dir, 'delta.cfg')]

    for filename in DEFAULT_CONFIG_FILES:
        if os.path.exists(filename):
            config.load(filename)

__load_initial_config()
