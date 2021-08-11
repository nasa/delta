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
Various helpful loss functions.
"""

import numpy as np

import tensorflow as tf
import tensorflow.keras.losses #pylint: disable=no-name-in-module
import tensorflow.keras.backend as K #pylint: disable=no-name-in-module

from delta.config import config
from delta.config.extensions import register_loss
from delta.ml.config_parser import loss_from_dict

def ms_ssim(y_true, y_pred):
    """
    `tf.image.ssim_multiscale` as a loss function.
    """
    return 1.0 - tf.image.ssim_multiscale(y_true, y_pred, 4.0)

def ms_ssim_mse(y_true, y_pred):
    """
    Sum of MS-SSIM and Mean Squared Error.
    """
    return ms_ssim(y_true, y_pred) + K.mean(K.mean(tensorflow.keras.losses.MSE(y_true, y_pred), -1), -1)

# from https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (
                K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth + K.epsilon())

def dice_loss(y_true, y_pred):
    """
    Dice coefficient as a loss function.
    """
    return 1 - dice_coef(y_true, y_pred)

class MappedLoss(tf.keras.losses.Loss): #pylint: disable=abstract-method
    def __init__(self, mapping, name=None):
        """
        This is a base class for losses when the labels of the input images do not match the labels
        output by the network. For example, if one class in the labels should be ignored, or two
        classes in the label should map to the same output, or one label should be treated as a probability
        between two classes. It applies a transform to the output labels and then applies the loss function.

        Note that the transform is applied after preprocessing (labels in the config will be transformed to 0-n
        in order, and nodata will be n+1).

        Parameters
        ----------
        mapping
            One of:
             * A list with transforms, where the first entry is what to transform the first label, to etc., i.e.,
               [1, 0] will swap the order of two labels.
             * A dictionary with classes mapped to transformed values. Classes can be referenced by name or by
               number (see `delta.imagery.imagery_config.ClassesConfig.class_id` for class formats).
        name: Optional[str]
            Optional name for the loss function.
        """
        super().__init__(name=name)
        self._nodata_classes = []
        if isinstance(mapping, list):
            map_list = mapping
            # replace nodata
            for (i, me) in enumerate(map_list):
                if me == 'nodata':
                    j = 0
                    while map_list[i] == 'nodata':
                        if j == len(map_list):
                            raise ValueError('All mapping entries are nodata.')
                        if map_list[j] != 'nodata':
                            map_list[i] = map_list[j]
                            break
                        j += 1
                    self._nodata_classes.append(i)
        else:
            # automatically set nodata to 0 (even if there is none it's fine)
            entry = mapping[next(iter(mapping))]
            if np.isscalar(entry):
                map_list = np.zeros((len(config.dataset.classes) + 1,))
            else:
                map_list = np.zeros((len(config.dataset.classes) + 1, len(entry)))
            assert len(mapping) == len(config.dataset.classes), 'Must specify all classes in loss mapping.'
            for k in mapping:
                i = config.dataset.classes.class_id(k)
                if isinstance(mapping[k], (int, float)):
                    map_list[i] = mapping[k]
                elif mapping[k] == 'nodata':
                    self._nodata_classes.append(i)
                else:
                    assert len(mapping[k]) == map_list.shape[1], 'Mapping entry wrong length.'
                    map_list[i, :] = np.asarray(mapping[k])
        self._lookup = tf.constant(map_list, dtype=tf.float32)

    # makes nodata labels 0 in predictions
    def preprocess(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)

        true_convert = tf.gather(self._lookup, y_true, axis=None)
        nodata_value = config.dataset.classes.class_id('nodata')
        nodata = (y_true == nodata_value)

        # ignore additional nodata classes
        for c in self._nodata_classes:
            nodata = tf.logical_or(nodata, y_true == c)

        while len(nodata.shape) < len(y_pred.shape):
            nodata = tf.expand_dims(nodata, -1)

        # zero all nodata entries
        y_pred = y_pred * tf.cast(tf.logical_not(nodata), tf.float32)

        true_convert = tf.cast(tf.logical_not(nodata), tf.float32) * true_convert
        return (true_convert, y_pred)

class MappedCategoricalCrossentropy(MappedLoss):
    """
    `MappedLoss` for categorical_crossentropy.
    """
    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        (y_true, y_pred) = self.preprocess(y_true, y_pred)
        return tensorflow.keras.losses.categorical_crossentropy(y_true, y_pred)

class MappedBinaryCrossentropy(MappedLoss):
    """
    `MappedLoss` for binary_crossentropy.
    """
    def call(self, y_true, y_pred):
        (y_true, y_pred) = self.preprocess(y_true, y_pred)
        return tensorflow.keras.losses.binary_crossentropy(y_true, y_pred)

class MappedDiceLoss(MappedLoss):
    """
    `MappedLoss` for `dice_loss`.
    """
    def call(self, y_true, y_pred):
        (y_true, y_pred) = self.preprocess(y_true, y_pred)
        return dice_loss(y_true, y_pred)

class MappedMsssim(MappedLoss):
    """
    `MappedLoss` for `ms_ssim`.
    """
    def call(self, y_true, y_pred):
        (y_true, y_pred) = self.preprocess(y_true, y_pred)
        return ms_ssim(y_true, y_pred)

class MappedDiceBceMsssim(MappedLoss):
    """
    `MappedLoss` for sum of `ms_ssim`, `dice_loss`, and `binary_crossentropy`.
    """
    def call(self, y_true, y_pred):
        (y_true, y_pred) = self.preprocess(y_true, y_pred)

        dice = dice_loss(y_true, y_pred)
        bce = tensorflow.keras.losses.binary_crossentropy(y_true, y_pred)
        msssim = ms_ssim(y_true, y_pred) # / tf.cast(tf.size(y_true), tf.float32)
        msssim = tf.expand_dims(tf.expand_dims(msssim, -1), -1)

        return dice + bce + msssim

class MappedLossSum(MappedLoss):
    """
    `MappedLoss` for sum of any loss functions.
    """
    def __init__(self, mapping, name=None, losses=None, weights=None):
        """
        Parameters
        ----------
        losses: List[Union[str, dict]]
            List of loss functions to add.
        weights: Union[List[float], None]
            Optional list of weights for the corresponding loss functions.
        """
        super().__init__(mapping, name=name)
        self._losses = list(map(loss_from_dict, losses))
        if weights is None:
            weights = [1] * len(losses)
        self._weights = weights

    def _get_loss(self, i, y_true, y_pred):
        l = self._losses[i](y_true, y_pred)
        while len(l.shape) < 3:
            l = tf.expand_dims(l, -1)
        return self._weights[i] * l

    def call(self, y_true, y_pred):
        (y_true, y_pred) = self.preprocess(y_true, y_pred)

        total = self._get_loss(0, y_true, y_pred)
        for i in range(1, len(self._losses)):
            total += self._get_loss(i, y_true, y_pred)

        return total

register_loss('ms_ssim', ms_ssim)
register_loss('ms_ssim_mse', ms_ssim_mse)
register_loss('dice', dice_loss)
register_loss('MappedCategoricalCrossentropy', MappedCategoricalCrossentropy)
register_loss('MappedBinaryCrossentropy', MappedBinaryCrossentropy)
register_loss('MappedDice', MappedDiceLoss)
register_loss('MappedMsssim', MappedMsssim)
register_loss('MappedDiceBceMsssim', MappedDiceBceMsssim)
register_loss('MappedLossSum', MappedLossSum)
