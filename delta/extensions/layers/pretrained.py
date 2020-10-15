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
import tensorflow.keras.models

from delta.ml.train import DeltaLayer

class Pretrained(DeltaLayer):
    def __init__(self, filename, encoding_layer, trainable=False, **kwargs):
        '''
        Loads a pretrained model and extracts the enocoding layers.
        '''
        super().__init__(**kwargs)
        assert filename is not None, 'Did not specify pre-trained model.'
        assert encoding_layer is not None, 'Did not specify encoding layer point.'

        self._filename = filename
        self._encoding_layer = encoding_layer
        self.trainable = trainable

        temp_model = tensorflow.keras.models.load_model(filename, compile=False)

        output_layers = []
        if isinstance(encoding_layer, int):
            break_point = lambda x, y: x == encoding_layer
        elif isinstance(encoding_layer, str):
            break_point = lambda x, y: y.name == encoding_layer

        for idx, l in enumerate(temp_model.layers):
            output_layers.append(l)
            output_layers[-1].trainable = trainable
            if break_point(idx, l):
                break
        #self._layers = tensorflow.keras.models.Sequential(output_layers, **kwargs)
        self.layers = output_layers
        self.input_spec = self.layers[0].input_spec

    def get_config(self):
        config = super().get_config()
        config.update({'filename': self._filename})
        config.update({'encoding_layer': self._encoding_layer})
        return config

    def call(self, inputs, **_):
        x = inputs
        for l in self.layers:
            x = l(x)
        return x

    def shape(self):
        return tf.TensorShape(self.layers[0].input_shape[0])
