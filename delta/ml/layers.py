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
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import Callback

class DeltaLayer(Layer):
    # optionally return a Keras callback
    def callback(self): # pylint:disable=no-self-use
        return None

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

ALL_LAYERS = {
    'GaussianSample' : GaussianSample,
    'Pretrained' : Pretrained
}
