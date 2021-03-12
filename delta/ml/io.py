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
Functions for IO specific to ML.
"""

import h5py
import numpy as np
import tensorflow.keras.backend as K

from delta.config import config

def save_model(model, filename):
    """
    Save a model. Includes DELTA configuration.

    Parameters
    ----------
    model: tensorflow.keras.models.Model
        The model to save.
    filename: str
        Output filename.
    """
    model.save(filename, save_format='h5')
    with h5py.File(filename, 'r+') as f:
        f.attrs['delta'] = config.export()

def print_layer(l):
    """
    Print a layer to stdout.

    l: tensorflow.keras.layers.Layer:
        The layer to print.
    """
    s = "{:<25}".format(l.name) + ' ' + '{:<20}'.format(str(l.input_shape)) + \
        ' -> ' + '{:<20}'.format(str(l.output_shape))
    c = l.get_config()
    if 'strides' in c:
        s += ' s: ' + '{:<10}'.format(str(c['strides']))
    if 'kernel_size' in c:
        s += ' ks: ' + str(c['kernel_size'])
    print(s)

def print_network(a, tile_shape=None):
    """
    Print a model to stdout.

    a: tensorflow.keras.models.Model:
        The model to print.
    tile_shape: Optional[Tuple[int, int]]
        If specified, print layer output sizes (necessary for FCN only).
    """
    for l in a.layers:
        print_layer(l)
    in_shape = a.layers[0].input_shape[0]
    if tile_shape is not None:
        in_shape = (in_shape[0], tile_shape[0], tile_shape[1], in_shape[3])
    out_shape = a.compute_output_shape(in_shape)
    print('Size: ' + str(in_shape[1:]) + ' --> ' + str(out_shape[1:]))
    if out_shape[1] is not None and out_shape[2] is not None:
        print('Compression Rate - ', out_shape[1] * out_shape[2] * out_shape[3] /
                                     (in_shape[1] * in_shape[2] * in_shape[3]))
    print('Layers - ', len(a.layers))
    print('Trainable Parameters - ', np.sum([K.count_params(w) for w in a.trainable_weights]))
