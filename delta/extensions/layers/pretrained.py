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
import tensorflow
import tensorflow.keras.models

from delta.config.extensions import register_layer

class InputSelectLayer(tensorflow.keras.layers.Layer):
    def __init__(self, arg_number, **kwargs):
        super().__init__(**kwargs)
        self._arg = arg_number
    def call(self, inputs, **kwargs):
        return inputs[self._arg]
    def get_config(self):
        return {'arg_number' : self._arg}

def pretrained(filename, encoding_layer, trainable=True, **kwargs):
    """
    Creates pre-trained layer from an existing model file.
    Only works with sequential models.
    """
    model = tensorflow.keras.models.load_model(filename, compile=False)

    if isinstance(encoding_layer, int):
        break_point = lambda x, y: x == encoding_layer
    elif isinstance(encoding_layer, str):
        break_point = lambda x, y: y.name == encoding_layer

    output_layers = []
    for idx, l in enumerate(model.layers):
        if not isinstance(l, tensorflow.keras.layers.BatchNormalization):
            l.trainable = trainable
        output_layers.append(l)
        if break_point(idx, l):
            break

    new_model = tensorflow.keras.models.Model(model.inputs, [l.output for l in output_layers], **kwargs)

    layers_dict = {}
    for (i, l) in enumerate(output_layers):
        layers_dict[l.name] = InputSelectLayer(i)

    def call(*inputs):
        result = new_model(inputs)
        output = (InputSelectLayer(len(layers_dict)-1)(result), {k : v(result) for k, v in layers_dict.items()})
        return output
    return call

register_layer('InputSelectLayer', InputSelectLayer)
register_layer('Pretrained', pretrained)
