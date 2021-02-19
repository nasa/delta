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
DELTA specific network layers.
"""

import tensorflow as tf
import tensorflow.keras.layers
import tensorflow.keras.backend as K

from delta.config.extensions import register_layer

# If layers inherit from callback as well we add them automatically on fit
class RepeatedGlobalAveragePooling2D(tensorflow.keras.layers.Layer):
    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, **_):
        ones = tf.fill(tf.shape(x)[:-1], 1.0)
        ones = tf.expand_dims(ones, -1)
        mean = K.mean(x, axis=[1, 2])
        mean = tf.expand_dims(mean, 1)
        mean = tf.expand_dims(mean, 1)
        return mean * ones

register_layer('RepeatedGlobalAveragePooling2D', RepeatedGlobalAveragePooling2D)