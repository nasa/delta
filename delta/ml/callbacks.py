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

import sys
from typing import List

import tensorflow
import tensorflow.keras.callbacks

from delta.config import config

class SetTrainable(tensorflow.keras.callbacks.Callback):
    def __init__(self, layer_name, epoch, make_trainable=True):
        super().__init__()
        self._layer_name = layer_name
        self._epoch = epoch
        self._make_trainable = make_trainable

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self._epoch:
            self.model.get_layer(self._layer_name).trainable = self._make_trainable

def ExponentialLRScheduler(start_epoch=10, multiplier=0.95):
    def schedule(epoch, lr):
        if epoch < start_epoch:
            return lr
        return multiplier * lr
    return tensorflow.keras.callbacks.LearningRateScheduler(schedule)

def callback_from_dict(callback_dict) -> tensorflow.keras.callbacks.Callback:
    '''
    Constructs a callback object from a dictionary.
    '''
    assert len(callback_dict.keys()) == 1, f'Error: Callback has more than one type {callback_dict.keys()}'

    layer_type = next(iter(callback_dict.keys()))
    callback_class = getattr(sys.modules[__name__], layer_type, None)
    if callback_class is None:
        callback_class = getattr(tensorflow.keras.callbacks, layer_type, None)
    if callback_dict[layer_type] is None:
        callback_dict[layer_type] = {}
    return callback_class(**callback_dict[layer_type])

def config_callbacks() -> List[tensorflow.keras.callbacks.Callback]:
    '''
    Iterates over the list of callbacks specified in the config file, which is part of the training specification.
    '''
    if not config.train.callbacks() is None:
        return [callback_from_dict(callback) for callback in config.train.callbacks()]
    return []
