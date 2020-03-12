"""
DELTA specific network layers.
"""

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
        super(GaussianSample, self).__init__(**kwargs)
        self._use_kl_loss = kl_loss
        self._kl_enabled = K.variable(0.0, name=self.name + ':kl_enabled')
        self.trainable = False

    def get_config(self):
        config = super(GaussianSample, self).get_config()
        config.update({'kl_loss': self._use_kl_loss})
        return config

    def callback(self):
        kl_enabled = self._kl_enabled
        class GaussianSampleCallback(Callback):
            def on_epoch_begin(self, epoch, _): # pylint:disable=no-self-use
                if epoch > 0:
                    K.set_value(kl_enabled, 1.0)
        return GaussianSampleCallback()

    def call(self, inputs):
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
            kl_loss /= batch * dim[0] * dim[1] * dim[2]

            kl_loss *= self._kl_enabled

            self.add_loss(kl_loss)
            self.add_metric(kl_loss, aggregation='mean', name=self.name + '_kl_loss')

        return result

ALL_LAYERS = {
    'GaussianSample' : GaussianSample
}
