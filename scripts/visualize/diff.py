#!/usr/bin/python3

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

# Creates a difference image between two images.

import sys

import numpy as np

from delta.extensions.sources.tiff import TiffImage, TiffWriter
from delta.imagery import rectangle

assert len(sys.argv) == 3, 'Please specify two tiff files of the same size.'

img1 = TiffImage(sys.argv[1])
img2 = TiffImage(sys.argv[2])

output_image = TiffWriter('diff.tiff')
output_image.initialize((img1.width(), img1.height(), 3), np.uint8, img1.metadata())

assert img1.width()== img2.width() and img1.height() == img2.height() and \
       img1.num_bands() == img2.num_bands(), 'Images must be same size.'

def callback_function(roi, data, _):
    data2 = img2.read(roi)
    diff = np.mean((data - data2) ** 2, axis=-1)
    diff = np.uint8(np.clip(diff * 128.0, 0.0, 255.0))
    out = np.stack([diff, diff, diff], axis=-1)
    output_image.write(out, roi.min_x, roi.min_y)

input_bounds = rectangle.Rectangle(0, 0, width=img1.width(), height=img1.height())
output_rois = input_bounds.make_tile_rois((2048, 2048), include_partials=True)[0]

img1.process_rois(output_rois, callback_function, show_progress=True)

output_image.close()
