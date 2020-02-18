from abc import ABC, abstractmethod
import numpy as np

import tensorflow as tf

from delta.imagery import rectangle

#pylint: disable=unsubscriptable-object
# Pylint was barfing lines 32 and 76. See relevant bug report
# https://github.com/PyCQA/pylint/issues/1498

class Predictor(ABC):
    """
    Abstract class to run prediction for an image given a model.
    """
    def __init__(self, model, show_progress=False):
        self._model = model
        self._show_progress = show_progress

    @abstractmethod
    def _initialize(self, shape, label):
        """
        Called at the start of a new prediction.
        The output shape and the label image are passed as inputs.
        """

    @abstractmethod
    def _process_block(self, pred_image, x, y, labels):
        """
        Processes a predicted block. The predictions are in pred_image,
        (sx, sy) is the starting coordinates of the block, and the corresponding labels
        if available are passed as labels.
        """

    @abstractmethod
    def output(self):
        """
        Returns the predicted output.
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
        """Like predict but returns (predicted image, error image, percent correct)."""
        TILE_SIZE = 256
        net_input_shape = self._model.get_input_shape_at(0)[1:]
        net_output_shape = self._model.get_output_shape_at(0)[1:]
        offset_r = -net_input_shape[0] + net_output_shape[0]
        offset_c = -net_input_shape[1] + net_output_shape[1]
        block_size_x = net_input_shape[0] * (TILE_SIZE // net_input_shape[0])
        block_size_y = net_input_shape[1] * (TILE_SIZE // net_input_shape[1])

        # Set up the output image
        if not input_bounds:
            input_bounds = rectangle.Rectangle(0, 0, width=image.width(), height=image.height())

        self._initialize((input_bounds.width() + offset_r, input_bounds.height() + offset_c), label)

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

        image.process_rois(output_rois, callback_function, show_progress=self._show_progress)
        return self.output()

class LabelPredictor(Predictor):
    """
    Predicts integer labels for an image.
    """
    def __init__(self, model, show_progress=False, probabilities=False):
        super(LabelPredictor, self).__init__(model, show_progress)
        self._probabilities = probabilities
        self._errors = None
        self._buffer = None
        self._confusion_matrix = None
        self._num_classes = None

    def _initialize(self, shape, label):
        net_output_shape = self._model.get_output_shape_at(0)[1:]
        self._num_classes = net_output_shape[-1]
        if label:
            self._errors = np.zeros(shape, dtype=np.bool)
            self._confusion_matrix = np.zeros((self._num_classes, self._num_classes), dtype=np.int32)
        else:
            self._errors = None
            self._confusion_matrix = None
        self._buffer = np.zeros(shape, dtype=np.float32)

    def _process_block(self, pred_image, x, y, labels):
        if self._probabilities:
            self._buffer[x : x + pred_image.shape[0], y : y + pred_image.shape[1], :] = pred_image
        pred_image = np.argmax(pred_image, axis=2)

        if not self._probabilities:
            self._buffer[x : x + pred_image.shape[0], y : y + pred_image.shape[1]] = pred_image

        if labels is not None:
            self._errors[x : x + pred_image.shape[0], y : y + pred_image.shape[1]] = labels != pred_image
            cm = tf.math.confusion_matrix(np.ndarray.flatten(labels),
                                          np.ndarray.flatten(pred_image),
                                          self._num_classes)
            self._confusion_matrix[:, :] += cm

    def output(self):
        return self._buffer

    def confusion_matrix(self):
        return self._confusion_matrix

    def errors(self):
        return self._errors

class ImagePredictor(Predictor):
    """
    Predicts an image from an image.
    """
    def __init__(self, model, show_progress=False):
        super(ImagePredictor, self).__init__(model, show_progress)
        self._errors = None
        self._buffer = None

    def _initialize(self, shape, label):
        net_output_shape = self._model.get_output_shape_at(0)[1:]
        if label:
            self._errors = np.zeros(shape + (net_output_shape[-1], ), dtype=np.float32)
        else:
            self._errors = None
        self._buffer = np.zeros(shape + (net_output_shape[-1], ), dtype=np.float32)

    def _process_block(self, pred_image, x, y, labels):
        self._buffer[x : x + pred_image.shape[0], y : y + pred_image.shape[1], :] = pred_image

        if labels is not None:
            self._errors[x : x + pred_image.shape[0], y : y + pred_image.shape[1], :] = np.abs(labels - pred_image)

    def output(self):
        return self._buffer

    def errors(self):
        return self._errors
