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
from delta.config.extensions import image_reader, preprocess_function
from . import disk_folder_cache


class ImageSet:
    """
    Specifies a set of image files.

    The images can be accessed by using the `ImageSet` as an iterable.
    """
    def __init__(self, images, image_type, preprocess=None, nodata_value=None):
        """
        Parameters
        ----------
        images: Iterator[str]
            Image filenames
        image_type: str
            The image type as a string (i.e., tiff, worldview, landsat). Must have
            been previously registered with `delta.config.extensions.register_image_reader`.
        preprocess: Callable
            Optional preprocessing function to apply to the image
            following the signature in `delta.imagery.delta_image.DeltaImage.set_preprocess`.
        nodata_value: image dtype
            A no data value for pixels to disregard
        """
        self._images = images
        self._image_type = image_type
        self._preprocess = preprocess
        self._nodata_value = nodata_value

    def type(self):
        """
        Returns
        -------
        str:
            The type of the image
        """
        return self._image_type
    def preprocess(self):
        """
        Returns
        -------
        Callable:
            The preprocessing function
        """
        return self._preprocess
    def nodata_value(self):
        """
        Returns
        -------
        image dtype:
            Value of pixels to disregard.
        """
        return self._nodata_value

    def set_nodata_value(self, nodata):
        """
        Set the pixel value to disregard.

        Parameters
        ----------
        nodata: image dtype
            The pixel value to set as nodata
        """
        self._nodata_value = nodata

    def load(self, index):
        """
        Loads the image of the given index.

        Parameters
        ----------
        index: int
            Index of the image to load.

        Returns
        -------
        `delta.imagery.delta_image.DeltaImage`:
            The image
        """
        img = image_reader(self.type())(self[index], self.nodata_value())
        if self._preprocess:
            img.set_preprocess(self._preprocess)
        return img

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
                        'npy' : '.npy',
                        'sentinel1' : '.zip'}

def __extension(conf):
    if conf['extension'] == 'default':
        return __DEFAULT_EXTENSIONS.get(conf['type'])
    return conf['extension']

def __find_images(conf, matching_images=None, matching_conf=None):
    '''
    Find the images specified by a given configuration, returning a list of images.
    If matching_images and matching_conf are specified, we find the labels matching these images.
    '''
    images = []
    if conf['type'] not in __DEFAULT_EXTENSIONS:
        raise ValueError('Unexpected image type %s.' % (conf['type']))

    if conf['files']:
        assert conf['file_list'] is None and conf['directory'] is None, 'Only one image specification allowed.'
        images = conf['files']
        for (i, im) in enumerate(images):
            images[i] = os.path.normpath(im)
    elif conf['file_list']:
        assert conf['directory'] is None, 'Only one image specification allowed.'
        with open(conf['file_list'], 'r') as f:
            for line in f:
                images.append(os.path.normpath(line.strip()))
    elif conf['directory']:
        extension = __extension(conf)
        if not os.path.exists(conf['directory']):
            raise ValueError('Supplied images directory %s does not exist.' % (conf['directory']))
        if matching_images is None:
            for root, _, filenames in os.walk(conf['directory'],
                                              followlinks=True):
                for filename in filenames:
                    if filename.endswith(extension):
                        images.append(os.path.join(root, filename))
        else:
            # find matching labels
            for m in matching_images:
                rel_path   = os.path.relpath(m, matching_conf['directory'])
                label_path = os.path.join(conf['directory'], rel_path)
                if matching_conf['directory'] is None:
                    images.append(os.path.splitext(label_path)[0] + extension)
                else:
                    # if custom extension, remove it
                    label_path = label_path[:-len(__extension(matching_conf))]
                    images.append(label_path + extension)

    for img in images:
        if not os.path.exists(img):
            raise ValueError('Image file %s does not exist.' % (img))
    return images

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

    pre = images_comp.preprocess_function()
    imageset = ImageSet(images, images_dict['type'], pre, images_dict['nodata_value'])

    if (labels_dict['files'] is None) and (labels_dict['file_list'] is None) and (labels_dict['directory'] is None):
        return (imageset, None)

    labels = __find_images(labels_dict, images, images_dict)

    if len(labels) != len(images):
        raise ValueError('%d images found, but %d labels found.' % (len(images), len(labels)))

    labels_nodata = labels_dict['nodata_value']
    pre_orig = labels_comp.preprocess_function()
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
    """
    Configuration for image preprocessing.

    Expects a list of preprocessing functions registered
    with `delta.config.extensions.register_preprocess`.
    """
    def __init__(self):
        super().__init__()
        self._functions = []

    def _load_dict(self, d, base_dir):
        if d is None:
            self._functions = []
            return
        if not d:
            return
        self._functions = []
        assert isinstance(d, list), 'preprocess should be list of commands'
        for func in d:
            if isinstance(func, str):
                self._functions.append((func, {}))
            else:
                assert isinstance(func, dict), 'preprocess items must be strings or dicts'
                assert len(func) == 1, 'One preprocess item per list entry.'
                name = list(func.keys())[0]
                self._functions.append((name, func[name]))

    def function(self, image_type):
        """
        Parameters
        ----------
        image_type: str
            Type of the image
        Returns
        -------
        Callable:
            The specified preprocessing function to apply to the image.
        """
        prep = lambda data, _, dummy: data
        for (name, args) in self._functions:
            t = preprocess_function(name)
            assert t is not None, 'Preprocess function %s not found.' % (name)
            p = t(image_type=image_type, **args)
            def helper(cur, prev):
                return lambda data, roi, bands: cur(prev(data, roi, bands), roi, bands)
            prep = helper(p, prep)
        return prep

def _validate_paths(paths, base_dir):
    out = []
    for p in paths:
        out.append(validate_path(p, base_dir))
    return out

class ImageSetConfig(DeltaConfigComponent):
    """
    Configuration for a set of images.

    Used for images, labels, and validation images and labels.
    """
    def __init__(self, name=None):
        super().__init__()
        self.register_field('type', str, 'type', None, 'Image type.')
        self.register_field('files', list, None, _validate_paths, 'List of image files.')
        self.register_field('file_list', str, None, validate_path, 'File listing image files.')
        self.register_field('directory', str, None, validate_path, 'Directory of image files.')
        self.register_field('extension', str, None, None, 'Image file extension.')
        self.register_field('nodata_value', (float, int), None, None, 'Value of pixels to ignore.')

        if name:
            self.register_arg('type', '--' + name + '-type', name + '_type')
            self.register_arg('file_list', '--' + name + '-file-list', name + '_file_list')
            self.register_arg('directory', '--' + name + '-dir', name + '_directory')
            self.register_arg('extension', '--' + name + '-extension', name + '_extension')
        self.register_component(ImagePreprocessConfig(), 'preprocess')
        self._name = name

    def preprocess_function(self):
        """
        Returns
        -------
        Callable:
            Preprocessing function for the set of images.
        """
        return self._components['preprocess'].function(self._config_dict['type'])

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
            self._config_dict['directory'] = None
            self._config_dict['file_list'] = None

class LabelClass:
    """
    Label configuration.
    """
    def __init__(self, value, name=None, color=None, weight=None):
        """
        Parameters
        ----------
        value: int
            Pixel of the label
        name: str
            Name of the class to display
        color: int
            In visualizations, set the class to this RGB color.
        weight: float
            During training weight this class by this amount.
        """
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
        self.end_value = None

    def __repr__(self):
        return 'Color: ' + self.name

class ClassesConfig(DeltaConfigComponent):
    """
    Configuration for classes.

    Specify either a number of classes or list of classes with details.
    """
    def __init__(self):
        super().__init__()
        self._classes = []
        self._conversions = []

    def __iter__(self):
        return self._classes.__iter__()

    def __getitem__(self, key):
        return self._classes[key]

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
        elif isinstance(d, dict):
            for k in d:
                assert isinstance(k, int), 'Class label value must be int.'
                self._classes.append(LabelClass(k, str(d[k].get('name')),
                                                d[k].get('color'), d[k].get('weight')))
        else:
            raise ValueError('Expected classes to be an int or list in config, was ' + str(d))
        # make sure the order is consistent for same values, and create preprocessing function
        self._conversions = []
        self._classes = sorted(self._classes, key=lambda x: x.value)
        for (i, v) in enumerate(self._classes):
            if v.value != i:
                self._conversions.append(v.value)
            v.end_value = i

    def class_id(self, class_name):
        """
        Parameters
        ----------
        class_name: int or str
            Either the original pixel value in images (int) or the name (str) of a class.
            The special value 'nodata' will give the nodata class, if any.

        Returns
        -------
        int:
            the ID of the class in the labels after default image preprocessing (labels are arranged
            to a canonical order, with nodata always coming after them.)
        """
        if class_name == len(self._classes) or class_name == 'nodata':
            return len(self._classes)
        for (i, c) in enumerate(self._classes):
            if class_name in (c.value, c.name):
                return i
        raise ValueError('Class ' + str(class_name) + ' not found.')

    def weights(self):
        """
        Returns
        -------
        List[float]
            List of class weights for use in training, if specified.
        """
        weights = []
        for c in self._classes:
            if c.weight is not None:
                weights.append(c.weight)
        if not weights:
            return None
        assert len(weights) == len(self._classes), 'For class weights, either all or none must be specified.'
        return weights

    def classes_to_indices_func(self):
        """
        Returns
        -------
        Callable[[numpy.ndarray], numpy.ndarray]:
            Function to convert label image to canonical form
        """
        if not self._conversions:
            return None
        def convert(data):
            assert isinstance(data, np.ndarray)
            for (i, c) in enumerate(self._conversions):
                data[data == c] = i
            return data
        return convert

    def indices_to_classes_func(self):
        """
        Returns
        -------
        Callable[[numpy.ndarray], numpy.ndarray]:
            Reverse of `classes_to_indices_func`.
        """
        if not self._conversions:
            return None
        def convert(data):
            assert isinstance(data, np.ndarray)
            for (i, c) in reversed(list(enumerate(self._conversions))):
                data[data == i] = c
            return data
        return convert

class DatasetConfig(DeltaConfigComponent):
    """
    Configuration for a dataset.
    """
    def __init__(self):
        super().__init__('Dataset')
        self.register_component(ImageSetConfig('image'), 'images', '__image_comp')
        self.register_component(ImageSetConfig('label'), 'labels', '__label_comp')
        self.__images = None
        self.__labels = None
        self.register_component(ClassesConfig(), 'classes')

    def reset(self):
        super().reset()
        self.__images = None
        self.__labels = None

    def images(self) -> ImageSet:
        """
        Returns
        -------
        ImageSet:
            the training images
        """
        if self.__images is None:
            (self.__images, self.__labels) = load_images_labels(self._components['images'],
                                                                self._components['labels'],
                                                                self._components['classes'])
        return self.__images

    def labels(self) -> ImageSet:
        """
        Returns
        -------
        ImageSet:
            the label images
        """
        if self.__labels is None:
            (self.__images, self.__labels) = load_images_labels(self._components['images'],
                                                                self._components['labels'],
                                                                self._components['classes'])
        return self.__labels

class CacheConfig(DeltaConfigComponent):
    """
    Configuration for cache.
    """
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
        Returns
        -------
        `disk_folder_cache.DiskCache`:
            the object to manage the cache
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
    """
    Configuration for I/O.
    """
    def __init__(self):
        super().__init__('IO')
        self.register_field('threads', int, None, None, 'Number of threads to use.')
        self.register_field('tile_size', list, 'tile_size', _validate_tile_size,
                            'Size of an image tile to load in memory at once.')

        self.register_arg('threads', '--threads')

        self.register_component(CacheConfig(), 'cache')

    def threads(self):
        """
        Returns
        -------
        int:
            number of threads to use for I/O
        """
        if 'threads' in self._config_dict and self._config_dict['threads']:
            return self._config_dict['threads']
        return min(1, os.cpu_count() // 2)

def register():
    """
    Registers imagery config options with the global config manager.

    cmd_args enables command line options if set to true.
    """
    config.register_component(DatasetConfig(), 'dataset')
    config.register_component(IOConfig(), 'io')
