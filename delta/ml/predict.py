"""
Module to run prediction according to learned neural networks
on images.
"""

import os
from abc import ABC, abstractmethod
import numpy as np

import tensorflow as tf

from delta.imagery import rectangle
from delta.imagery.sources.tiff import TiffWriter, numpy_dtype_to_gdal_type

#pylint: disable=unsubscriptable-object
# Pylint was barfing lines 32 and 76. See relevant bug report
# https://github.com/PyCQA/pylint/issues/1498

_TILE_SIZE = 256

def _clean_delete(filename):
    if filename is None:
        return
    try:
        os.remove(filename)
    except OSError:
        pass

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
        net_input_shape = self._model.get_input_shape_at(0)[1:]
        net_output_shape = self._model.get_output_shape_at(0)[1:]

        assert net_input_shape[2] == data.shape[2],\
               'Model expects %d input channels, data has %d channels' % (net_input_shape[2], data.shape[2])

        out_shape = (data.shape[0] - net_input_shape[0] + net_output_shape[0],
                     data.shape[1] - net_input_shape[1] + net_output_shape[1])
        image = tf.convert_to_tensor(data)
        image = tf.expand_dims(image, 0)
        chunks = tf.image.extract_patches(image, [1, net_input_shape[0], net_input_shape[1], 1],
                                          [1, net_output_shape[0], net_output_shape[1], 1],
                                          [1, 1, 1, 1], padding='VALID')
        chunks = tf.reshape(chunks, (-1,) + net_input_shape)

        best = np.zeros((chunks.shape[0],) + net_output_shape, dtype=np.float32)
        ## TODO: other data types, configurable batch size
        BATCH_SIZE=1000
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
        net_input_shape = self._model.get_input_shape_at(0)[1:]
        net_output_shape = self._model.get_output_shape_at(0)[1:]
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
        super(LabelPredictor, self).__init__(model, show_progress)
        self._confusion_matrix = None
        self._num_classes = None
        self._output_image = output_image
        self._colormap = colormap
        self._prob_image = prob_image
        self._error_image = error_image
        self._error_colors = error_colors
        if self._output_image:
            assert self._colormap is not None, 'Must specify colormap.'
        if self._error_image:
            assert self._error_colors is not None, 'Must specify error_colors.'
        self._output = None
        self._prob_o = None
        self._errors = None

    def _initialize(self, shape, label, image):
        net_output_shape = self._model.get_output_shape_at(0)[1:]
        self._num_classes = net_output_shape[-1]
        if label:
            self._errors = np.zeros(shape, dtype=np.bool)
            self._confusion_matrix = np.zeros((self._num_classes, self._num_classes), dtype=np.int32)
        else:
            self._errors = None
            self._confusion_matrix = None
        if self._output_image:
            self._output = TiffWriter(self._output_image, shape[0], shape[1], num_bands=self._colormap.shape[1],
                                      data_type=numpy_dtype_to_gdal_type(self._colormap.dtype),
                                      metadata=image.metadata(),
                                      tile_width=_TILE_SIZE, tile_height=_TILE_SIZE)
        if self._prob_image:
            self._prob_o = TiffWriter(self._prob_image, shape[0], shape[1], num_bands=self._num_classes,
                                      data_type=numpy_dtype_to_gdal_type(np.float32), metadata=image.metadata(),
                                      tile_width=_TILE_SIZE, tile_height=_TILE_SIZE)
        if self._error_image:
            self._errors = TiffWriter(self._error_image, shape[0], shape[1], num_bands=self._num_classes,
                                      data_type=numpy_dtype_to_gdal_type(np.float32), metadata=image.metadata(),
                                      tile_width=_TILE_SIZE, tile_height=_TILE_SIZE)

    def _complete(self):
        if self._output:
            self._output.close()
        if self._prob_o:
            self._prob_o.close()
        if self._errors:
            self._errors.close()

    def _abort(self):
        self._complete()
        _clean_delete(self._output_image)
        _clean_delete(self._prob_image)
        _clean_delete(self._error_image)

    def _process_block(self, pred_image, x, y, labels):
        if self._prob_o is not None:
            self._prob_o.write_region(pred_image, x, y)
        pred_image = np.argmax(pred_image, axis=2)

        if self._output is not None:
            self._output.write_region(self._colormap[pred_image], x, y)

        if labels is not None:
            self._errors.write_region(labels != pred_image, x, y)
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
        Trains on model, outputs to output_image.

        transform is a tuple (function, output numpy type, number of bands) applied
        to the output image.
        """
        super(ImagePredictor, self).__init__(model, show_progress)
        self._output_image = output_image
        self._output = None
        self._transform = transform

    def _initialize(self, shape, label, image):
        net_output_shape = self._model.get_output_shape_at(0)[1:]
        if self._output_image is not None:
            dtype = np.float32 if self._transform is None else self._transform[1]
            bands = net_output_shape[-1] if self._transform is None else self._transform[2]
            self._output = TiffWriter(self._output_image, shape[0], shape[1], num_bands=bands,
                                      data_type=numpy_dtype_to_gdal_type(dtype), metadata=image.metadata(),
                                      tile_width=_TILE_SIZE, tile_height=_TILE_SIZE)

    def _complete(self):
        if self._output:
            self._output.close()

    def _abort(self):
        self._complete()
        _clean_delete(self._output_image)

    def _process_block(self, pred_image, x, y, labels):
        if self._output is not None:
            im = pred_image
            if self._transform is not None:
                im = self._transform[0](im)
            self._output.write_region(im, x, y)
