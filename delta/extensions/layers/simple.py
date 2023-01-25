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
Simple helpful layers.
"""

import tensorflow as tf
import tensorflow.keras.layers #pylint: disable=no-name-in-module
import tensorflow.keras.backend as K #pylint: disable=no-name-in-module

from delta.config.extensions import register_layer

class RepeatedGlobalAveragePooling2D(tensorflow.keras.layers.Layer):
    """
    Global average pooling in 2D for fully convolutional networks.

    Takes the global average over the entire input, and repeats
    it to return a tensor the same size as the input.
    """
    def compute_output_shape(self, input_shape): # pylint: disable=no-self-use
        return input_shape

    def call(self, inputs, **_): # pylint: disable=no-self-use,arguments-differ
        ones = tf.fill(tf.shape(inputs)[:-1], 1.0)
        ones = tf.expand_dims(ones, -1)
        mean = K.mean(inputs, axis=[1, 2])
        mean = tf.expand_dims(mean, 1)
        mean = tf.expand_dims(mean, 1)
        return mean * ones

class ReflectionPadding2D(tensorflow.keras.layers.Layer):
    """
    Add reflected padding of the given size surrounding the input.
    """
    def __init__(self, padding=(1, 1), **kwargs):
        super().__init__(**kwargs)
        self.padding = tuple(padding)

    def get_config(self):
        config = super().get_config()
        config.update({'padding': self.padding})
        return config

    def call(self, inputs, **_): # pylint: disable=arguments-differ
        w_pad,h_pad = self.padding
        return tf.pad(inputs, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

register_layer('RepeatedGlobalAveragePooling2D', RepeatedGlobalAveragePooling2D)
register_layer('ReflectionPadding2D', ReflectionPadding2D)
