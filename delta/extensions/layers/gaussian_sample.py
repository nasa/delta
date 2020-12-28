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

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

from delta.config.extensions import register_layer
from delta.ml.train import DeltaLayer

# If layers inherit from callback as well we add them automatically on fit
class GaussianSample(DeltaLayer):
    def __init__(self, kl_loss=True, **kwargs):
        super().__init__(**kwargs)
        self._use_kl_loss = kl_loss
        self._kl_enabled = K.variable(0.0, name=self.name + ':kl_enabled')
        self.trainable = False

    def get_config(self):
        config = super().get_config()
        config.update({'kl_loss': self._use_kl_loss})
        return config

    def callback(self):
        kl_enabled = self._kl_enabled
        class GaussianSampleCallback(Callback):
            def on_epoch_begin(self, epoch, _=None): # pylint:disable=no-self-use
                if epoch > 0:
                    K.set_value(kl_enabled, 1.0)
        return GaussianSampleCallback()

    def call(self, inputs, **_):
        mean, log_var = inputs
        batch = K.shape(mean)[0]
        dim = K.int_shape(mean)[1:]
        epsilon = K.random_normal(shape=(batch, ) + dim)
        result = mean + K.exp(0.5 * log_var) * epsilon

        if self._use_kl_loss:
            # this loss function makes the mean and variance match a Normal(0, 1) distribution
            kl_loss = K.square(mean) + K.exp(log_var) - 1 - log_var
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss = 0.5 * K.mean(kl_loss)

            # reduce relative weight compared to mean squared error
            kl_loss /= K.cast(batch * dim[0] * dim[1] * dim[2], dtype='float32')

            kl_loss *= self._kl_enabled

            self.add_loss(kl_loss)
            self.add_metric(kl_loss, aggregation='mean', name=self.name + '_kl_loss')

        return result

register_layer('GaussianSample', GaussianSample)
