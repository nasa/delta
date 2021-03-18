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
# pylint: disable=too-many-ancestors
"""
Various helpful loss functions.
"""

import tensorflow as tf
import tensorflow.keras.metrics

from delta.config import config
from delta.config.extensions import register_metric

class SparseMetric(tensorflow.keras.metrics.Metric): # pylint:disable=abstract-method # pragma: no cover
    """
    An abstract class for metrics applied to integer class labels,
    with networks that output one-hot encoding.
    """
    def __init__(self, label, class_id: int=None, name: str=None, binary: int=False):
        """
        Parameters
        ----------
        label
            A class identifier accepted by `delta.imagery.imagery_config.ClassesConfig.class_id`.
            Compared to valuse in the label image.
        class_id: Optional[int]
            For multi-class one-hot outputs, used if the output class ID is different than the
            one in the label image.
        name: str
            Metric name.
        binary: bool
            Use binary threshold (0.5) or argmax on one-hot encoding.
        """
        super().__init__(name=name)
        self._binary = binary
        self._label_id = config.dataset.classes.class_id(label)
        self._class_id = class_id if class_id is not None else self._label_id

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

class SparseRecall(SparseMetric): # pragma: no cover
    """
    Recall.
    """
    def __init__(self, label, class_id: int=None, name: str=None, binary: int=False):
        super().__init__(label, class_id, name, binary)
        self._total_class = self.add_weight('total_class', initializer='zeros')
        self._true_positives = self.add_weight('true_positives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None): #pylint: disable=unused-argument, arguments-differ
        y_true = tf.squeeze(y_true)
        right_class = tf.math.equal(y_true, self._label_id)
        if self._binary:
            y_pred = y_pred >= 0.5
            right_class_pred = tf.squeeze(y_pred)
        else:
            y_pred = tf.math.argmax(y_pred, axis=-1)
            right_class_pred = tf.math.equal(y_pred, self._class_id)
        total_class = tf.math.reduce_sum(tf.cast(right_class, tf.float32))
        self._total_class.assign_add(total_class)
        true_positives = tf.math.logical_and(right_class, right_class_pred)
        true_positives = tf.math.reduce_sum(tf.cast(true_positives, tf.float32))
        self._true_positives.assign_add(true_positives)

    def result(self):
        return tf.math.divide_no_nan(self._true_positives, self._total_class)

class SparsePrecision(SparseMetric): # pragma: no cover
    """
    Precision.
    """
    def __init__(self, label, class_id: int=None, name: str=None, binary: int=False):
        super().__init__(label, class_id, name, binary)
        self._total_class = self.add_weight('total_class', initializer='zeros')
        self._true_positives = self.add_weight('true_positives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None): #pylint: disable=unused-argument, arguments-differ
        y_true = tf.squeeze(y_true)
        right_class = tf.math.equal(y_true, self._label_id)
        if self._binary:
            y_pred = y_pred >= 0.5
            right_class_pred = tf.squeeze(y_pred)
        else:
            y_pred = tf.math.argmax(y_pred, axis=-1)
            right_class_pred = tf.math.equal(y_pred, self._class_id)

        total_class = tf.math.reduce_sum(tf.cast(right_class_pred, tf.float32))
        self._total_class.assign_add(total_class)
        true_positives = tf.math.logical_and(right_class, right_class_pred)
        true_positives = tf.math.reduce_sum(tf.cast(true_positives, tf.float32))
        self._true_positives.assign_add(true_positives)

    def result(self):
        return tf.math.divide_no_nan(self._true_positives, self._total_class)

class SparseBinaryAccuracy(SparseMetric): # pragma: no cover
    """
    Accuracy.
    """
    def __init__(self, label, name: str=None):
        super().__init__(label, label, name, False)
        self._nodata_id = config.dataset.classes.class_id('nodata')
        self._total = self.add_weight('total', initializer='zeros')
        self._correct = self.add_weight('correct', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None): #pylint: disable=unused-argument, arguments-differ
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)

        right_class = tf.math.equal(y_true, self._label_id)
        right_class_pred = y_pred >= 0.5
        true_positives = tf.math.logical_and(right_class, right_class_pred)
        false_negatives = tf.math.logical_and(tf.math.logical_not(right_class), tf.math.logical_not(right_class_pred))
        if self._nodata_id:
            valid = tf.math.not_equal(y_true, self._nodata_id)
            false_negatives = tf.math.logical_and(false_negatives, valid)
            total = tf.math.reduce_sum(tf.cast(valid, tf.float32))
        else:
            total = tf.size(y_true)

        true_positives = tf.math.reduce_sum(tf.cast(true_positives, tf.float32))
        false_negatives = tf.math.reduce_sum(tf.cast(false_negatives, tf.float32))
        self._correct.assign_add(true_positives + false_negatives)
        self._total.assign_add(total)

    def result(self):
        return tf.math.divide(self._correct, self._total)

register_metric('SparseRecall', SparseRecall)
register_metric('SparsePrecision', SparsePrecision)
register_metric('SparseBinaryAccuracy', SparseBinaryAccuracy)
