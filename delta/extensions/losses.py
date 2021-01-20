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
Various helpful loss functions.
"""

import numpy as np

import tensorflow as tf
import tensorflow.keras.losses
import tensorflow.keras.backend as K

from delta.config import config
from delta.config.extensions import register_loss

def ms_ssim(y_true, y_pred):
    return 1.0 - tf.image.ssim_multiscale(y_true, y_pred, 4.0)

def ms_ssim_mse(y_true, y_pred):
    return ms_ssim(y_true, y_pred) + K.mean(K.mean(tensorflow.keras.losses.MSE(y_true, y_pred), -1), -1)

class MappedCategoricalCrossentropy(tf.keras.losses.Loss):
    # this is cross entropy, but first replaces the labels with
    # a probability distribution from a lookup table
    def __init__(self, mapping):
        """
        Pass as argument either a list with probabilities for labels in order,
        or a dictionary with classes mapped to their probabilities.
        """
        super().__init__()
        if isinstance(mapping, list):
            map_list = mapping
        else:
            # automatically set nodata to 0 (even if there is none it's fine)
            entry = next(iter(mapping))
            if np.isscalar(entry):
                map_list = np.zeros((len(config.dataset.classes) + 1,))
            else:
                map_list = np.zeros((len(config.dataset.classes) + 1, len(entry)))
            assert len(mapping) == len(config.dataset.classes), 'Must specify all classes in loss mapping.'
            for k in mapping:
                i = config.dataset.classes.class_id(k)
                map_list[i] = mapping[k]
        self._lookup = tf.constant(map_list)
    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        true_convert = tf.gather(self._lookup, tf.cast(y_true, tf.int32), axis=None)
        return tensorflow.keras.losses.categorical_crossentropy(true_convert, y_pred)

register_loss('ms_ssim', ms_ssim)
register_loss('ms_ssim_mse', ms_ssim_mse)
register_loss('MappedCategoricalCrossentropy', MappedCategoricalCrossentropy)
