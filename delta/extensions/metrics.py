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

import tensorflow as tf
import tensorflow.keras.metrics

from delta.config import config
from delta.config.extensions import register_metric

class SparseRecall(tensorflow.keras.metrics.Metric): # pragma: no cover
    # this is cross entropy, but first replaces the labels with
    # a probability distribution from a lookup table
    def __init__(self, label, class_id=None, name=None):
        super().__init__(name=name)
        self._label_id = config.dataset.classes.class_id(label)
        self._class_id = class_id if class_id is not None else self._label_id
        self._total_class = self.add_weight('total_class', initializer='zeros')
        self._true_positives = self.add_weight('true_positives', initializer='zeros')

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None): #pylint: disable=unused-argument, arguments-differ
        y_true = tf.squeeze(y_true)
        y_pred = tf.math.argmax(y_pred, axis=-1)

        right_class = tf.math.equal(y_true, self._label_id)
        total_class = tf.math.reduce_sum(tf.cast(right_class, tf.float32))
        self._total_class.assign_add(total_class)
        true_positives = tf.math.logical_and(right_class, tf.math.equal(y_pred, self._class_id))
        true_positives = tf.math.reduce_sum(tf.cast(true_positives, tf.float32))
        self._true_positives.assign_add(true_positives)
        return self._total_class, self._true_positives

    def result(self):
        return tf.math.divide_no_nan(self._true_positives, self._total_class)

register_metric('SparseRecall', SparseRecall)
