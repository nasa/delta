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
Load images given configuration.
"""

from . import worldview, landsat, tiff, npy

_IMAGE_TYPES = {
        'worldview' : worldview.WorldviewImage,
        'landsat' : landsat.LandsatImage,
        'tiff' : tiff.TiffImage,
        'rgba' : tiff.RGBAImage,
        'npy' : npy.NumpyImage
}

def register_image_type(image_type, image_class):
    """
    Register a custom image type for use by DELTA.

    image_type is a string specified in config files.
    image_class is a custom class that extends
    `delta.iamge.sources.delta_image.DeltaImage`.
    """
    global _IMAGE_TYPES #pylint: disable=global-statement
    _IMAGE_TYPES[image_type] = image_class

def load(filename, image_type, preprocess=False):
    """
    Load an image of the appropriate type and parameters.
    """
    if image_type not in _IMAGE_TYPES:
        raise ValueError('Unexpected image_type %s.' % (image_type))
    img = _IMAGE_TYPES[image_type](filename)
    if preprocess:
        img.set_preprocess(preprocess)
    return img

def load_image(image_set, index):
    """
    Load the specified image in the ImageSet.
    """
    return load(image_set[index], image_set.type(), preprocess=image_set.preprocess())
