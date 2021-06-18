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
import tensorflow.keras.callbacks #pylint: disable=no-name-in-module

from delta.config.extensions import register_callback
from delta.ml.train import ContinueTrainingException

class SetTrainable(tensorflow.keras.callbacks.Callback):
    """
    Changes whether a given layer is trainable during training.

    This is useful for transfer learning, to do an initial training and then allow fine-tuning.
    """
    def __init__(self, layer_name: str, epoch: int, trainable: bool=True, learning_rate: float=None):
        """
        Parameters
        ----------
        layer_name: str
            The layer to modify.
        epoch: int
            The change will take place at the start of this epoch (the first epoch is 1).
        trainable: bool
            Whether the layer will be made trainable or not trainable.
        learning_rate: float
            Optionally change the learning rate as well.
        """
        super().__init__()
        self._layer_name = layer_name
        self._epoch = epoch - 1
        self._make_trainable = trainable
        self._lr = learning_rate
        self._triggered = False

    def on_epoch_begin(self, epoch, logs=None): # pylint: disable=unused-argument
        if epoch == self._epoch:
            if self._triggered:
                return
            self._triggered = True # don't repeat twice
            l = self.model.get_layer(self._layer_name)
            l.trainable = True
            # have to abort, recompile changed model, and continue training
            raise ContinueTrainingException(completed_epochs=epoch, recompile_model=True, learning_rate=self._lr)

def ExponentialLRScheduler(start_epoch: int=10, multiplier: float=0.95):
    """
    Schedule the learning rate exponentially.

    Parameters
    ----------
    start_epoch: int
        The epoch to begin.
    multiplier: float
        After `start_epoch`, multiply the learning rate by this amount each epoch.
    """
    def schedule(epoch, lr):
        if epoch < start_epoch:
            return lr
        return multiplier * lr
    return tensorflow.keras.callbacks.LearningRateScheduler(schedule)

register_callback('SetTrainable', SetTrainable)
register_callback('ExponentialLRScheduler', ExponentialLRScheduler)
