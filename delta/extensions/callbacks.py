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
Custom callbacks that come with DELTA.
"""

import tensorflow
import tensorflow.keras.callbacks

from delta.config import config
from delta.config.extensions import register_callback
from delta.ml.train import compile_model

class SetTrainable(tensorflow.keras.callbacks.Callback):
    def __init__(self, layer_name, epoch, trainable=True):
        super().__init__()
        self._layer_name = layer_name
        self._epoch = epoch
        self._make_trainable = trainable

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self._epoch:
            l = self.model.get_layer(self._layer_name)
            if isinstance(l, tensorflow.keras.Model):
                for a in l.layers:
                    # don't change batchnormalization, special case
                    if not isinstance(a, tensorflow.keras.layers.BatchNormalization):
                        a.trainable = self._make_trainable
            else:
                l.trainable = self._make_trainable
            compile_model(self.model, config.train.spec())

def ExponentialLRScheduler(start_epoch=10, multiplier=0.95):
    def schedule(epoch, lr):
        if epoch < start_epoch:
            return lr
        return multiplier * lr
    return tensorflow.keras.callbacks.LearningRateScheduler(schedule)

register_callback('SetTrainable', SetTrainable)
register_callback('ExponentialLRScheduler', ExponentialLRScheduler)
