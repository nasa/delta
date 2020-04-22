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
Specify a set of image files.
"""

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
