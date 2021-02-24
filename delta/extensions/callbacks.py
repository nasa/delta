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

from delta.config.extensions import register_callback
from delta.ml.train import ContinueTrainingException

class SetTrainable(tensorflow.keras.callbacks.Callback):
    def __init__(self, layer_name, epoch, trainable=True, learning_rate=None):
        super().__init__()
        self._layer_name = layer_name
        self._epoch = epoch - 1
        self._make_trainable = trainable
        self._lr = learning_rate
        self._triggered = False

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self._epoch:
            if self._triggered:
                return
            self._triggered = True # don't repeat twice
            l = self.model.get_layer(self._layer_name)
            l.trainable = True
            # have to abort, recompile changed model, and continue training
            raise ContinueTrainingException(completed_epochs=epoch, recompile_model=True, learning_rate=self._lr)

def ExponentialLRScheduler(start_epoch=10, multiplier=0.95):
    def schedule(epoch, lr):
        if epoch < start_epoch:
            return lr
        return multiplier * lr
    return tensorflow.keras.callbacks.LearningRateScheduler(schedule)

register_callback('SetTrainable', SetTrainable)
register_callback('ExponentialLRScheduler', ExponentialLRScheduler)
