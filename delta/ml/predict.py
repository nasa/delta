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
Module to run prediction according to learned neural networks
on images.
"""

from abc import ABC, abstractmethod
import numpy as np

import tensorflow as tf

from delta.imagery import rectangle
from delta.config import config

#pylint: disable=unsubscriptable-object
# Pylint was barfing lines 32 and 76. See relevant bug report
# https://github.com/PyCQA/pylint/issues/1498

_TILE_SIZE = 256

class Predictor(ABC):
    """
    Abstract class to run prediction for an image given a model.
    """
    def __init__(self, model, show_progress=False):
        self._model = model
        self._show_progress = show_progress

    @abstractmethod
    def _initialize(self, shape, label, image):
        """
        Called at the start of a new prediction.
        The output shape, label image, and image being read are passed as inputs.
        """

    def _complete(self):
        """Called to do any cleanup if needed."""

    def _abort(self):
        """Cancel the operation and cleanup neatly."""

    @abstractmethod
    def _process_block(self, pred_image, x, y, labels):
        """
        Processes a predicted block. The predictions are in pred_image,
        (sx, sy) is the starting coordinates of the block, and the corresponding labels
        if available are passed as labels.
        """

    def _predict_array(self, data):
        net_input_shape = self._model.input_shape[1:]
        net_output_shape = self._model.output_shape[1:]

        assert net_input_shape[2] == data.shape[2],\
               'Model expects %d input channels, data has %d channels' % (net_input_shape[2], data.shape[2])

        out_shape = (data.shape[0] - net_input_shape[0] + net_output_shape[0],
                     data.shape[1] - net_input_shape[1] + net_output_shape[1])
        out_type = self._model.get_output_at(0).dtype
        image = tf.convert_to_tensor(data)
        image = tf.expand_dims(image, 0)
        chunks = tf.image.extract_patches(image, [1, net_input_shape[0], net_input_shape[1], 1],
                                          [1, net_output_shape[0], net_output_shape[1], 1],
                                          [1, 1, 1, 1], padding='VALID')
        chunks = tf.reshape(chunks, (-1,) + net_input_shape)

        best = np.zeros((chunks.shape[0],) + net_output_shape, dtype=out_type.as_numpy_dtype)
        BATCH_SIZE = int(config.io.block_size_mb() * 1024 * 1024 / net_input_shape[0] / net_input_shape[1] /
                         net_input_shape[2] / out_type.size)
        assert BATCH_SIZE > 0, 'block_size_mb too small.'
        for i in range(0, chunks.shape[0], BATCH_SIZE):
            best[i:i+BATCH_SIZE] = self._model.predict_on_batch(chunks[i:i+BATCH_SIZE])

        retval = np.zeros(out_shape + (net_output_shape[-1],))
        for chunk_idx in range(0, best.shape[0]):
            r = (chunk_idx // (  out_shape[1] // net_output_shape[1])) * net_output_shape[0]
            c = (chunk_idx  % ( out_shape[1] // net_output_shape[1])) * net_output_shape[1]
            retval[r:r+net_output_shape[0],c:c+net_output_shape[1],:] = best[chunk_idx,:,:,:]

        return retval

    def predict(self, image, label=None, input_bounds=None):
        """
        Runs the model on `image`, comparing the results to `label` if specified.
        Results are limited to `input_bounds`. Returns output, the meaning of which
        depends on the subclass.
        """
        net_input_shape = self._model.input_shape[1:]
        net_output_shape = self._model.output_shape[1:]
        offset_r = -net_input_shape[0] + net_output_shape[0]
        offset_c = -net_input_shape[1] + net_output_shape[1]
        block_size_x = net_input_shape[0] * (_TILE_SIZE // net_input_shape[0])
        block_size_y = net_input_shape[1] * (_TILE_SIZE // net_input_shape[1])

        # Set up the output image
        if not input_bounds:
            input_bounds = rectangle.Rectangle(0, 0, width=image.width(), height=image.height())

        self._initialize((input_bounds.width() + offset_r, input_bounds.height() + offset_c), label, image)

        def callback_function(roi, data):
            pred_image = self._predict_array(data)

            block_x = (roi.min_x - input_bounds.min_x)
            block_y = (roi.min_y - input_bounds.min_y)
            (sx, sy) = (block_x , block_y)

            labels = None
            if label:
                label_x = roi.min_x + (roi.width() - pred_image.shape[0]) // 2
                label_y = roi.min_y + (roi.height() - pred_image.shape[1]) // 2
                label_roi = rectangle.Rectangle(label_x, label_y,
                                                label_x + pred_image.shape[0], label_y + pred_image.shape[1])
                labels = np.squeeze(label.read(label_roi))

            self._process_block(pred_image, sx, sy, labels)

        output_rois = input_bounds.make_tile_rois(block_size_x - offset_r, block_size_y - offset_c,
                                                  include_partials=False, overlap_amount=-offset_r)

        try:
            image.process_rois(output_rois, callback_function, show_progress=self._show_progress)
        except KeyboardInterrupt:
            self._abort()
            raise

        return self._complete()
class LabelPredictor(Predictor):
    """
    Predicts integer labels for an image.
    """
    def __init__(self, model, output_image=None, show_progress=False,
                 colormap=None, prob_image=None, error_image=None, error_colors=None):
        """
        output_image, prob_image, and error_image are all DeltaImageWriter's.
        colormap and error_colors are all numpy arrays mapping classes to colors.
        """
        super(LabelPredictor, self).__init__(model, show_progress)
        self._confusion_matrix = None
        self._num_classes = None
        self._output_image = output_image
        self._colormap = colormap
        self._prob_image = prob_image
        self._error_image = error_image
        self._error_colors = error_colors
        if self._error_image:
            assert self._error_colors is not None, 'Must specify error_colors.'
        self._output = None
        self._prob_o = None
        self._errors = None

    def _initialize(self, shape, label, image):
        net_output_shape = self._model.output_shape[1:]
        self._num_classes = net_output_shape[-1]
        if label:
            self._errors = np.zeros(shape, dtype=np.bool)
            self._confusion_matrix = np.zeros((self._num_classes, self._num_classes), dtype=np.int32)
        else:
            self._errors = None
            self._confusion_matrix = None
        if self._output_image:
            if self._colormap is not None:
                self._output_image.initialize((shape[0], shape[1], self._colormap.shape[1]),
                                              self._colormap.dtype, image.metadata())
            else:
                self._output_image.initialize((shape[0], shape[1], 1), np.int32, image.metadata())
        if self._prob_image:
            self._prob_image.initialize((shape[0], shape[1], self._num_classes), np.float32, image.metadata())
        if self._error_image:
            self._error_image.initialize((shape[0], shape[1], self._error_colors.shape[1]),
                                         self._error_colors.dtype, image.metadata())

    def _complete(self):
        if self._output_image:
            self._output_image.close()
        if self._prob_image:
            self._prob_image.close()
        if self._error_image:
            self._error_image.close()

    def _abort(self):
        self._complete()
        if self._output_image is not None:
            self._output_image.abort()
        if self._prob_image is not None:
            self._prob_image.abort()
        if self._error_image is not None:
            self._error_image.abort()

    def _process_block(self, pred_image, x, y, labels):
        if self._prob_image is not None:
            self._prob_image.write(pred_image, x, y)
        pred_image = np.argmax(pred_image, axis=2)

        if self._output_image is not None:
            if self._colormap is not None:
                self._output_image.write(self._colormap[pred_image], x, y)
            else:
                self._output_image.write(pred_image, x, y)

        if labels is not None:
            self._error_image.write(self._error_colors[(labels != pred_image).astype(int)], x, y)
            cm = tf.math.confusion_matrix(np.ndarray.flatten(labels),
                                          np.ndarray.flatten(pred_image),
                                          self._num_classes)
            self._confusion_matrix[:, :] += cm

    def confusion_matrix(self):
        """
        Returns a matrix counting true labels matched to predicted labels.
        """
        return self._confusion_matrix

class ImagePredictor(Predictor):
    """
    Predicts an image from an image.
    """
    def __init__(self, model, output_image=None, show_progress=False, transform=None):
        """
        Trains on model, outputs to output_image, which is a DeltaImageWriter.

        transform is a tuple (function, output numpy type, number of bands) applied
        to the output image.
        """
        super(ImagePredictor, self).__init__(model, show_progress)
        self._output_image = output_image
        self._output = None
        self._transform = transform

    def _initialize(self, shape, label, image):
        net_output_shape = self._model.output_shape[1:]
        if self._output_image is not None:
            dtype = np.float32 if self._transform is None else self._transform[1]
            bands = net_output_shape[-1] if self._transform is None else self._transform[2]
            self._output_image.initialize((shape[0], shape[1], bands), dtype, image.metadata())

    def _complete(self):
        if self._output_image is not None:
            self._output_image.close()

    def _abort(self):
        self._complete()
        if self._output_image is not None:
            self._output_image.abort()

    def _process_block(self, pred_image, x, y, labels):
        if self._output_image is not None:
            im = pred_image
            if self._transform is not None:
                im = self._transform[0](im)
            self._output_image.write(im, x, y)
