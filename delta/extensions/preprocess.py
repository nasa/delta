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
#pylint:disable=unused-argument
"""
Various helpful preprocessing functions.
"""
import numpy as np

from delta.config.extensions import register_preprocess

__DEFAULT_SCALE_FACTORS = {'tiff' : 1024.0,
                           'worldview' : 1024.0,
                           'landsat' : 120.0,
                           'npy' : None,
                           'sentinel1' : None}

def scale(image_type, factor='default'):
    if factor == 'default':
        factor = __DEFAULT_SCALE_FACTORS[image_type]
    factor = np.float32(factor)
    return (lambda data, _, dummy: data / factor)

def offset(image_type, factor):
    factor = np.float32(factor)
    return lambda data, _, dummy: data + factor

def clip(image_type, bounds):
    if isinstance(bounds, list):
        assert len(bounds) == 2, 'Bounds must have two items.'
    else:
        bounds = (bounds, bounds)
    bounds = (np.float32(bounds[0]), np.float32(bounds[1]))
    return lambda data, _, dummy: np.clip(data, bounds[0], bounds[1])

def cbrt(image_type):
    return lambda data, _, dummy: np.cbrt(data)
def sqrt(image_type):
    return lambda data, _, dummy: np.sqrt(data)

def gauss_mult_noise(image_type, stddev):
    return lambda data, _, dummy: data * np.random.normal(1.0, stddev, data.shape)

def substitute(image_type, mapping):
    return lambda data, _, dummy: np.take(mapping, data)

register_preprocess('scale', scale)
register_preprocess('offset', offset)
register_preprocess('clip', clip)
register_preprocess('sqrt', sqrt)
register_preprocess('cbrt', cbrt)
register_preprocess('gauss_mult_noise', gauss_mult_noise)
register_preprocess('substitute', substitute)
