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

    def call(self, inputs, **_):
        ones = tf.fill(tf.shape(inputs)[:-1], 1.0)
        ones = tf.expand_dims(ones, -1)
        mean = K.mean(inputs, axis=[1, 2])
        mean = tf.expand_dims(mean, 1)
        mean = tf.expand_dims(mean, 1)
        return mean * ones

class ReflectionPadding2D(tensorflow.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [tensorflow.keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({'padding': self.padding})
        return config

#    def get_output_shape_for(self, s):
#        """ If you are using "channels_last" configuration"""
#        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        #tf.print(tf.shape(x), h_pad, w_pad)
        result = tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
        #tf.print(tf.shape(result))
        return result

register_layer('RepeatedGlobalAveragePooling2D', RepeatedGlobalAveragePooling2D)
register_layer('ReflectionPadding2D', ReflectionPadding2D)
