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
Configuration options specific to imagery.
"""
import os
import os.path
import numpy as np

import appdirs

from delta.config import config, DeltaConfigComponent, validate_path, validate_positive
from . import disk_folder_cache


class ImageSet:
    """
    Specifies a set of image files.

    The images can be accessed by using the `ImageSet` as an iterable.
    """
    def __init__(self, images, image_type, preprocess=None, nodata_value=None):
        """
        The parameters for the constructor are:

         * An iterable of image filenames `images`
         * The image type (i.e., tiff, worldview, landsat) `image_type`
         * An optional preprocessing function to apply to the image,
           following the signature in `delta.imagery.sources.delta_image.DeltaImage.set_process`.
         * A `nodata_value` for pixels to disregard
        """
        self._images = images
        self._image_type = image_type
        self._preprocess = preprocess
        self._nodata_value = nodata_value

    def type(self):
        """
        The type of the image (used by `delta.imagery.sources.loader`).
        """
        return self._image_type
    def preprocess(self):
        """
        Return the preprocessing function.
        """
        return self._preprocess
    def nodata_value(self):
        """
        Value of pixels to disregard.
        """
        return self._nodata_value
    def __len__(self):
        return len(self._images)
    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError('Index %s out of range.' % (index))
        return self._images[index]
    def __iter__(self):
        return self._images.__iter__()

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
    except ValueError as e:
        raise ValueError('Scale factor is %s, must be a float.' % (f)) from e

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

def load_images_labels(images_comp, labels_comp, classes_comp):
    '''
    Takes two configuration subsections and returns (image set, label set). Also takes classes
    configuration to apply preprocessing function to labels.
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
    imageset = ImageSet(images, images_dict['type'], pre, images_dict['nodata_value'])

    if (labels_dict['files'] is None) and (labels_dict['file_list'] is None) and (labels_dict['directory'] is None):
        return (imageset, None)

    labels = __find_images(labels_dict, images, images_dict)

    if len(labels) != len(images):
        raise ValueError('%d images found, but %d labels found.' % (len(images), len(labels)))

    labels_nodata = labels_dict['nodata_value']
    pre_orig = __preprocess_function(labels_comp)
    # we shift the label images to always be 0...n[+1], Class 1, Class 2, ... Class N, [nodata]
    def class_shift(data, _, dummy):
        if pre_orig is not None:
            data = pre_orig(data, _, dummy)
        # set any nodata values to be past the expected range
        if labels_nodata is not None:
            nodata_indices = (data == labels_nodata)
        conv = classes_comp.classes_to_indices_func()
        if conv is not None:
            data = conv(data)
        if labels_nodata is not None:
            data[nodata_indices] = len(classes_comp)
        return data
    return (imageset, ImageSet(labels, labels_dict['type'],
                               class_shift, len(classes_comp) if labels_nodata is not None else None))

class ImagePreprocessConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self.register_field('enabled', bool, 'enabled', None, 'Turn on preprocessing.')
        self.register_field('scale_factor', (float, str), 'scale_factor', None, 'Image scale factor.')

def _validate_paths(paths, base_dir):
    out = []
    for p in paths:
        out.append(validate_path(p, base_dir))
    return out

class ImageSetConfig(DeltaConfigComponent):
    def __init__(self, name=None):
        super().__init__()
        self.register_field('type', str, 'type', None, 'Image type.')
        self.register_field('files', list, None, _validate_paths, 'List of image files.')
        self.register_field('file_list', list, None, validate_path, 'File listing image files.')
        self.register_field('directory', str, None, validate_path, 'Directory of image files.')
        self.register_field('extension', str, None, None, 'Image file extension.')
        self.register_field('nodata_value', (float, int), None, None, 'Value of pixels to ignore.')

        if name:
            self.register_arg('type', '--' + name + '-type')
            self.register_arg('file_list', '--' + name + '-file-list')
            self.register_arg('directory', '--' + name + '-dir')
            self.register_arg('extension', '--' + name + '-extension')
        self.register_component(ImagePreprocessConfig(), 'preprocess')
        self._name = name

    def setup_arg_parser(self, parser, components = None) -> None:
        if self._name is None:
            return
        super().setup_arg_parser(parser, components)
        parser.add_argument("--" + self._name, dest=self._name, required=False,
                            help="Specify a single image file.")

    def parse_args(self, options):
        if self._name is None:
            return
        super().parse_args(options)
        if hasattr(options, self._name) and getattr(options, self._name) is not None:
            self._config_dict['files'] = [getattr(options, self._name)]

class LabelClass:
    def __init__(self, value, name=None, color=None, weight=None):
        color_order = [0x1f77b4, 0xff7f0e, 0x2ca02c, 0xd62728, 0x9467bd, 0x8c564b, \
                       0xe377c2, 0x7f7f7f, 0xbcbd22, 0x17becf]
        if name is None:
            name = 'Class ' + str(value)
        if color is None:
            color = color_order[value] if value < len(color_order) else 0
        self.value = value
        self.name = name
        self.color = color
        self.weight = weight

    def __repr__(self):
        return 'Color: ' + self.name

class ClassesConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self._classes = []
        self._conversions = []

    def __iter__(self):
        return self._classes.__iter__()

    def __len__(self):
        return len(self._classes)

    # overwrite model entirely if updated (don't want combined layers from multiple files)
    def _load_dict(self, d : dict, base_dir):
        if not d:
            return
        self._config_dict = d
        self._classes = []
        if isinstance(d, int):
            for i in range(d):
                self._classes.append(LabelClass(i))
        elif isinstance(d, list):
            for (i, c) in enumerate(d):
                if isinstance(c, int): # just pixel value
                    self._classes.append(LabelClass(i))
                else:
                    keys = c.keys()
                    assert len(keys) == 1, 'Dict should have name of pixel value.'
                    k = next(iter(keys))
                    assert isinstance(k, int), 'Class label value must be int.'
                    inner_dict = c[k]
                    self._classes.append(LabelClass(k, str(inner_dict.get('name')),
                                                    inner_dict.get('color'), inner_dict.get('weight')))
        else:
            raise ValueError('Expected classes to be an int or list in config, was ' + str(d))
        # make sure the order is consistent for same values, and create preprocessing function
        self._conversions = []
        self._classes = sorted(self._classes, key=lambda x: x.value)
        for (i, v) in enumerate(self._classes):
            if v.value != i:
                self._conversions.append(v.value)

    def weights(self):
        weights = []
        for c in self._classes:
            if c.weight is not None:
                weights.append(c.weight)
        if not weights:
            return None
        assert len(weights) == len(self._classes), 'For class weights, either all or none must be specified.'
        return weights

    def classes_to_indices_func(self):
        if not self._conversions:
            return None
        def convert(data):
            assert isinstance(data, np.ndarray)
            for (i, c) in enumerate(self._conversions):
                data[data == c] = i
            return data
        return convert

    def indices_to_classes_func(self):
        if not self._conversions:
            return None
        def convert(data):
            assert isinstance(data, np.ndarray)
            for (i, c) in reversed(list(enumerate(self._conversions))):
                data[data == i] = c
            return data
        return convert

class DatasetConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__('Dataset')
        self.register_component(ImageSetConfig('image'), 'images', '__image_comp')
        self.register_component(ImageSetConfig('label'), 'labels', '__label_comp')
        self.__images = None
        self.__labels = None
        self.register_field('log_folder', str, 'log_folder', validate_path,
                            'Directory where dataset progress is recorded.')
        self.register_component(ClassesConfig(), 'classes')

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
                                                                self._components['classes'])
        return self.__images

    def labels(self) -> ImageSet:
        """
        Returns the label images.
        """
        if self.__labels is None:
            (self.__images, self.__labels) = load_images_labels(self._components['images'],
                                                                self._components['labels'],
                                                                self._components['classes'])
        return self.__labels

class CacheConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self.register_field('dir', str, None, validate_path, 'Cache directory.')
        self.register_field('limit', int, None, validate_positive, 'Number of items to cache.')

        self._cache_manager = None

    def reset(self):
        super().reset()
        self._cache_manager = None

    def manager(self) -> disk_folder_cache.DiskCache:
        """
        Returns the disk cache object to manage the cache.
        """
        if self._cache_manager is None:
            # Auto-populating defaults here is a workaround so small tools can skip the full
            # command line config setup.  Could be improved!
            if 'dir' not in self._config_dict:
                self._config_dict['dir'] = 'default'
            if 'limit' not in self._config_dict:
                self._config_dict['limit'] = 8
            cdir = self._config_dict['dir']
            if cdir == 'default':
                cdir = appdirs.AppDirs('delta', 'nasa').user_cache_dir
            self._cache_manager = disk_folder_cache.DiskCache(cdir, self._config_dict['limit'])
        return self._cache_manager

def _validate_tile_size(size, _):
    assert len(size) == 2, 'Size must have two components.'
    assert isinstance(size[0], int) and isinstance(size[1], int), 'Size must be integer.'
    assert size[0] > 0 and size[1] > 1, 'Size must be positive.'
    return size

class IOConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self.register_field('threads', int, 'threads', None, 'Number of threads to use.')
        self.register_field('tile_size', list, 'tile_size', _validate_tile_size,
                            'Size of an image tile to load in memory at once.')
        self.register_field('interleave_images', int, 'interleave_images', validate_positive,
                            'Number of images to interleave at a time when training.')
        self.register_field('resume_cutoff', int, 'resume_cutoff', None,
                            'When resuming a dataset, skip images where we have read this many tiles.')

        self.register_field('stop_on_input_error', bool, 'stop_on_input_error', None,
                            'If false, skip past bad input images.')
        self.register_arg('stop_on_input_error', '--bypass-input-errors',
                          action='store_const', const=False, type=None)
        self.register_arg('stop_on_input_error', '--stop-on-input-error',
                          action='store_const', const=True, type=None)

        self.register_arg('threads', '--threads')

        self.register_component(CacheConfig(), 'cache')

def register():
    """
    Registers imagery config options with the global config manager.

    cmd_args enables command line options if set to true.
    """
    config.register_component(DatasetConfig(), 'dataset')
    config.register_component(IOConfig(), 'io')

