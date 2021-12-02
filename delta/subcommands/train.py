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
Train a neural network.
"""

import sys
import time

# import logging
# logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
from tensorflow.keras import mixed_precision

from delta.config import config
from delta.imagery import imagery_dataset
from delta.ml.train import train
from delta.ml.config_parser import config_model
from delta.ml.io import save_model, load_model


# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

def mixed_policy_device_compatible():

    # gpu check logic taken from https://github.com/keras-team/keras/blob/70d7d07bd186b929d81f7a8ceafff5d78d8bd701/keras/mixed_precision/device_compatibility_check.py # pylint: disable=line-too-long
    gpus = tf.config.list_physical_devices('GPU')
    gpu_details_list = [tf.config.experimental.get_device_details(g) for g in gpus]

    supported_device_strs = []
    unsupported_device_strs = []
    for details in gpu_details_list:
        name = details.get('device_name', 'Unknown GPU')
        cc = details.get('compute_capability')
        if cc:
            device_str = '%s, compute capability %s.%s' % (name, cc[0], cc[1])
            if cc >= (7, 0):
                supported_device_strs.append(device_str)
            else:
                unsupported_device_strs.append(device_str)
        else:
            unsupported_device_strs.append(
                name + ', no compute capability (probably not an Nvidia GPU)')

    if unsupported_device_strs or not supported_device_strs:
        return False
    # else mixed policy is compatible
    return True


def main(options):
    if mixed_policy_device_compatible() and not config.train.disable_mixed_precision():
        mixed_precision.set_global_policy('mixed_float16')
        print('Tensorflow Mixed Precision is enabled. This improves training performance on compatible GPUs. '
              'However certain precautions should be taken and several additional changes can be made to improve '
              'performance further. Details: https://www.tensorflow.org/guide/mixed_precision#summary')

    images = config.dataset.images()
    if not images:
        print('No images specified.', file=sys.stderr)
        return 1

    img = images.load(0)
    model = config_model(img.num_bands())
    if options.resume is not None:
        temp_model = load_model(options.resume)
    else:
        # this one is not built with proper scope, just used to get input and output shapes
        temp_model = model()

    start_time = time.time()
    tile_size = config.io.tile_size()
    tile_overlap = None
    stride = config.train.spec().stride

    # compute input and output sizes
    if temp_model.input_shape[1] is None:
        in_shape = None
        out_shape = temp_model.compute_output_shape((0, tile_size[0], tile_size[1], temp_model.input_shape[3]))
        out_shape = out_shape[1:3]
        tile_overlap = (tile_size[0] - out_shape[0], tile_size[1] - out_shape[1])
    else:
        in_shape = temp_model.input_shape[1:3]
        out_shape = temp_model.output_shape[1:3]

    if options.autoencoder:
        ids = imagery_dataset.AutoencoderDataset(images, in_shape, tile_shape=tile_size,
                                                 tile_overlap=tile_overlap, stride=stride,
                                                 max_rand_offset=config.train.spec().max_tile_offset)
    else:
        labels = config.dataset.labels()
        if not labels:
            print('No labels specified.', file=sys.stderr)
            return 1
        ids = imagery_dataset.ImageryDataset(images, labels, out_shape, in_shape,
                                             tile_shape=tile_size, tile_overlap=tile_overlap,
                                             stride=stride, max_rand_offset=config.train.spec().max_tile_offset)

    assert temp_model.input_shape[1] == temp_model.input_shape[2], 'Must have square chunks in model.'
    assert temp_model.input_shape[3] == ids.num_bands(), 'Model takes wrong number of bands.'
    tf.keras.backend.clear_session()

    # Try to have the internal model format we use match the output model format
    internal_model_extension = '.savedmodel'
    if options.model and ('.h5' in options.model):
        internal_model_extension = '.h5'
    try:
        model, _ = train(model, ids, config.train.spec(), options.resume, internal_model_extension)

        if options.model is not None:
            save_model(model, options.model)
    except KeyboardInterrupt:
        print('Training cancelled.')

    stop_time = time.time()
    print('Elapsed time = ', stop_time-start_time)
    return 0
