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

def load_images_labels(images_comp, labels_comp):
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
    imageset = ImageSet(images, images_dict['type'], pre, images_dict['nodata_value'])

    if (labels_dict['files'] is None) and (labels_dict['file_list'] is None) and (labels_dict['directory'] is None):
        return (imageset, None)

    labels = __find_images(labels_dict, images, images_dict)

    if len(labels) != len(images):
        raise ValueError('%d images found, but %d labels found.' % (len(images), len(labels)))

    pre = __preprocess_function(labels_comp)
    return (imageset, ImageSet(labels, labels_dict['type'],
                               pre, labels_dict['nodata_value']))

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
        self.register_field('nodata_value', float, None, None, 'Value of pixels to ignore.')

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

class DatasetConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__('Dataset')
        self.register_component(ImageSetConfig('image'), 'images', '__image_comp')
        self.register_component(ImageSetConfig('label'), 'labels', '__label_comp')
        self.__images = None
        self.__labels = None
        self.register_field('log_folder', str, 'log_folder', validate_path,
                            'Directory where dataset progress is recorded.')

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
                                                                self._components['labels'])
        return self.__images

    def labels(self) -> ImageSet:
        """
        Returns the label images.
        """
        if self.__labels is None:
            (self.__images, self.__labels) = load_images_labels(self._components['images'],
                                                                self._components['labels'])
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

class IOConfig(DeltaConfigComponent):
    def __init__(self):
        super().__init__()
        self.register_field('threads', int, 'threads', None, 'Number of threads to use.')
        self.register_field('block_size_mb', int, 'block_size_mb', validate_positive,
                            'Size of an image block to load in memory at once.')
        self.register_field('interleave_images', int, 'interleave_images', validate_positive,
                            'Number of images to interleave at a time when training.')
        self.register_field('tile_ratio', float, 'tile_ratio', validate_positive,
                            'Width to height ratio of blocks to load in images.')
        self.register_field('resume_cutoff', int, 'resume_cutoff', None,
                            'When resuming a dataset, skip images where we have read this many tiles.')

        self.register_arg('threads', '--threads')
        self.register_arg('block_size_mb', '--block-size-mb')
        self.register_arg('tile_ratio', '--tile-ratio')

        self.register_component(CacheConfig(), 'cache')

def register():
    """
    Registers imagery config options with the global config manager.

    cmd_args enables command line options if set to true.
    """
    config.register_component(DatasetConfig(), 'dataset')
    config.register_component(IOConfig(), 'io')
