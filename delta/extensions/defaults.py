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
Module to install all extensions that come with DELTA by default.
"""

from delta.config.extensions import register_extension, register_image_reader, register_image_writer

from .sources import tiff
from .sources import landsat
from .sources import npy
from .sources import worldview
from .sources import sentinel1

def initialize():
    """
    Register all default extensions.
    """
    register_extension('delta.extensions.callbacks')

    register_extension('delta.extensions.layers.pretrained')
    register_extension('delta.extensions.layers.gaussian_sample')
    register_extension('delta.extensions.layers.efficientnet')
    register_extension('delta.extensions.layers.simple')
    register_extension('delta.extensions.layers.deeplabv3plus')

    register_extension('delta.extensions.augmentations')
    register_extension('delta.extensions.losses')
    register_extension('delta.extensions.metrics')
    register_extension('delta.extensions.preprocess')

    register_image_reader('tiff', tiff.TiffImage)
    register_image_reader('npy', npy.NumpyImage)
    register_image_reader('landsat', landsat.LandsatImage)
    register_image_reader('worldview', worldview.WorldviewImage)
    register_image_reader('sentinel1', sentinel1.Sentinel1Image)

    register_image_writer('tiff', tiff.TiffWriter)
    register_image_writer('npy', npy.NumpyWriter)
