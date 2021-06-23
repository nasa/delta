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
Use a pretrained model inside another network.
"""
from typing import List, Optional
import tensorflow
import tensorflow.keras.models #pylint: disable=no-name-in-module

from delta.ml.io import load_model
from delta.config.extensions import register_layer

class InputSelectLayer(tensorflow.keras.layers.Layer):
    """
    A layer that takes any number of inputs, and returns a given one.
    """
    def __init__(self, arg_number, **kwargs):
        """
        Parameters
        ----------
        arg_number: int
            The index of the input to select.
        """
        super().__init__(**kwargs)
        self._arg = arg_number
    def call(self, inputs, **kwargs): #pylint: disable=unused-argument
        return inputs[self._arg]
    def get_config(self):
        return {'arg_number' : self._arg}

def _model_to_output_layers(model, break_point, trainable):
    output_layers = []
    for idx, l in enumerate(model.layers):
        if not isinstance(l, tensorflow.keras.layers.BatchNormalization):
            l.trainable = trainable
        if isinstance(l, tensorflow.keras.models.Model): # assumes sequential
            output_layers.extend(_model_to_output_layers(l, break_point, trainable))
        else:
            output_layers.append(l)
        if break_point(idx, l):
            break
    return output_layers

def pretrained(filename, encoding_layer, outputs: Optional[List[str]]=None, trainable: bool=True,
               training: bool=True, **kwargs):
    """
    Creates pre-trained layer from an existing model file.
    Only works with sequential models. This was quite tricky to get right with tensorflow.

    Parameters
    ----------
    filename: str
        Model file to load.
    encoding_layer: str
        Name of the layer to stop at.
    outputs: Optional[List[str]]
        List of names of output layers that may be used later in the model.
        Only layers listed here will be accessible as inputs to other layers, in the form
        this_layer_name/internal_name. (internal_name must be included in outputs to do so)
    trainable: bool
        Whether to update weights during training for this layer.
    training: bool
        Standard tensorflow option, used for batch norm layers.
    """
    model = load_model(filename)

    if isinstance(encoding_layer, int):
        break_point = lambda x, y: x == encoding_layer
    elif isinstance(encoding_layer, str):
        break_point = lambda x, y: y.name == encoding_layer

    output_layers = _model_to_output_layers(model, break_point, trainable)

    output_tensors = []
    cur = model.inputs[0]
    old_to_new = {}
    old_to_new[cur.ref()] = cur
    for l in output_layers:
        if isinstance(l, tensorflow.keras.layers.InputLayer):
            old_to_new[l.output.ref()] = cur
            output_tensors.append(cur)
            continue
        if isinstance(l.input, list):
            inputs = [old_to_new[t.ref()] for t in l.input]
        else:
            inputs = old_to_new[l.input.ref()]
        cur = l(inputs)
        old_to_new[l.output.ref()] = cur
        output_tensors.append(cur)
    new_model = tensorflow.keras.models.Model(model.inputs, output_tensors, **kwargs)

    layers_dict = {}
    if outputs:
        for (i, l) in enumerate(output_layers):
            if l.name not in outputs:
                continue
            layers_dict[l.name] = InputSelectLayer(i)

    def call(*inputs):
        result = new_model(inputs, training=training)
        output = (InputSelectLayer(len(output_layers)-1)(result), {k : v(result) for k, v in layers_dict.items()})
        return output
    return call

register_layer('InputSelectLayer', InputSelectLayer)
register_layer('Pretrained', pretrained)
