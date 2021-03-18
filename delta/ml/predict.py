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
import math
import numpy as np

import tensorflow as tf

from delta.imagery import rectangle

class Predictor(ABC):
    """
    Abstract class to run prediction for an image given a model.
    """
    def __init__(self, model, tile_shape=None, show_progress=False):
        self._model = model
        self._show_progress = show_progress
        self._tile_shape = tile_shape

    @abstractmethod
    def _initialize(self, shape, image, label=None):
        """
        Called at the start of a new prediction.

        Parameters
        ----------
        shape: (int, int)
            The final output shape from the network.
        image: delta.imagery.delta_image.DeltaImage
            The image to classify.
        label: delta.imagery.delta_image.DeltaImage
            The label image, if provided (otherwise None).
        """

    def _complete(self):
        """Called to do any cleanup if needed."""

    def _abort(self):
        """Cancel the operation and cleanup neatly."""

    @abstractmethod
    def _process_block(self, pred_image: np.ndarray, x: int, y: int, labels: np.ndarray, label_nodata):
        """
        Processes a predicted block. Must be overriden in subclasses.

        Parameters
        ----------
        pred_image: numpy.ndarray
            Output of model for a block of the image.
        x: int
            Top-left x coordinate of block.
        y: int
            Top-left y coordinate of block.
        labels: numpy.ndarray
            Labels (or None if not available) for same block as `pred_image`.
        label_nodata: dtype of labels
            Pixel value for nodata (or None).
        """

    def _predict_array(self, data: np.ndarray, image_nodata_value):
        """
        Runs model on data.

        Parameters
        ----------
        data: np.ndarray
            Block of image to apply the model to.
        image_nodata_value: dtype of data
            Nodata value in image. If given, nodata values are
            replaced with nans in output.

        Returns
        -------
        np.ndarray:
            Result of applying model to data.
        """
        net_input_shape = self._model.input_shape[1:]
        net_output_shape = self._model.output_shape[1:]

        image = tf.convert_to_tensor(data)
        image = tf.expand_dims(image, 0)

        assert net_input_shape[2] == data.shape[2],\
               'Model expects %d input channels, data has %d channels' % (net_input_shape[2], data.shape[2])

        # supports variable input size, just toss everything in
        if net_input_shape[0] is None and net_input_shape[1] is None:
            result = np.squeeze(self._model.predict_on_batch(image))
            if image_nodata_value is not None:
                x0 = (data.shape[0] - result.shape[0]) // 2
                y0 = (data.shape[1] - result.shape[1]) // 2
                invalid = (data if len(data.shape) == 2 else \
                          data[:, :, 0])[x0:x0 + result.shape[0], y0:y0 + result.shape[1]] == image_nodata_value
                if len(result.shape) == 2:
                    result[invalid] = math.nan
                else:
                    result[invalid, :] = math.nan
            return result

        out_shape = (data.shape[0] - net_input_shape[0] + net_output_shape[0],
                     data.shape[1] - net_input_shape[1] + net_output_shape[1])
        out_type = tf.dtypes.as_dtype(self._model.dtype)
        chunks = tf.image.extract_patches(image, [1, net_input_shape[0], net_input_shape[1], 1],
                                          [1, net_output_shape[0], net_output_shape[1], 1],
                                          [1, 1, 1, 1], padding='VALID')
        chunks = tf.reshape(chunks, (-1,) + net_input_shape)

        best = np.zeros((chunks.shape[0],) + net_output_shape, dtype=out_type.as_numpy_dtype)
        # do 8 MB at a time... this is arbitrary, may want to change in future
        BATCH_SIZE = max(1, int(8 * 1024 * 1024 / net_input_shape[0] / net_input_shape[1] /
                            net_input_shape[2] / out_type.size))
        for i in range(0, chunks.shape[0], BATCH_SIZE):
            best[i:i+BATCH_SIZE] = self._model.predict_on_batch(chunks[i:i+BATCH_SIZE])

        retval = np.zeros(out_shape + (net_output_shape[-1],))
        for chunk_idx in range(0, best.shape[0]):
            r = (chunk_idx // (  out_shape[1] // net_output_shape[1])) * net_output_shape[0]
            c = (chunk_idx  % ( out_shape[1] // net_output_shape[1])) * net_output_shape[1]
            retval[r:r+net_output_shape[0],c:c+net_output_shape[1],:] = best[chunk_idx,:,:,:]

        if image_nodata_value is not None:
            ox = (data.shape[0] - out_shape[0]) // 2
            oy = (data.shape[1] - out_shape[1]) // 2
            output_slice = data[ox:-ox, oy:-oy, 0]
            retval[output_slice == image_nodata_value] = math.nan

        return retval

    def predict(self, image, label=None, input_bounds=None, overlap=(0, 0)):
        """
        Runs the model on an image. The behavior is specific to the subclass.

        Parameters
        ----------
        image: delta.imagery.delta_image.DeltaImage
            Image to evalute.
        label: delta.imagery.delta_image.DeltaImage
            Optional label to compare to.
        input_bounds: delta.imagery.rectangle.Rectangle
            If specified, only evaluate the given portion of the image.
        overlap: (int, int)
            `predict` evaluates the image by selecting tiles, dependent on the tile_shape
            provided in the subclass. If an overlap is specified, the tiles will be overlapped
            by the given amounts in the x and y directions. Subclasses may select or interpolate
            to favor tile interior pixels for improved classification.

        Returns
        -------
        The result of the `_complete` function, which depends on the sublcass.
        """
        net_input_shape = self._model.input_shape[1:]
        net_output_shape = self._model.output_shape[1:]

        # Set up the output image
        if not input_bounds:
            input_bounds = rectangle.Rectangle(0, 0, width=image.width(), height=image.height())
        output_shape = (input_bounds.width(), input_bounds.height())

        ts = self._tile_shape if self._tile_shape else (image.width(), image.height())
        if net_input_shape[0] is None and net_input_shape[1] is None:
            assert net_output_shape[0] is None and net_output_shape[1] is None
            out_shape = self._model.compute_output_shape((0, ts[0], ts[1], net_input_shape[2]))
            tiles = input_bounds.make_tile_rois(ts, include_partials=False,
                                                overlap_shape=(ts[0] - out_shape[1] + overlap[0],
                                                               ts[1] - out_shape[2] + overlap[1]),
                                                partials_overlap=True)

        else:
            offset_r = -net_input_shape[0] + net_output_shape[0] + overlap[0]
            offset_c = -net_input_shape[1] + net_output_shape[1] + overlap[1]
            output_shape = (output_shape[0] + offset_r, output_shape[1] + offset_c)
            block_size_x = net_input_shape[0] * max(1, ts[0] // net_input_shape[0])
            block_size_y = net_input_shape[1] * max(1, ts[1] // net_input_shape[1])
            tiles = input_bounds.make_tile_rois((block_size_x - offset_r, block_size_y - offset_c),
                                                include_partials=False, overlap_shape=(-offset_r, -offset_c))

        self._initialize(output_shape, image, label)

        label_nodata = label.nodata_value() if label else None

        def callback_function(roi, data):
            pred_image = self._predict_array(data, image.nodata_value())

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

            tl = [0, 0]
            tl = (overlap[0] // 2 if block_x > 0 else 0, overlap[1] // 2 if block_y > 0 else 0)
            br = (roi.max_x - roi.min_x, roi.max_y - roi.min_y)
            br = (br[0] - (overlap[0] // 2 if roi.max_x < input_bounds.max_x else 0),
                  br[1] - (overlap[1] // 2 if roi.max_x < input_bounds.max_x else 0))
            if len(pred_image.shape) == 2:
                input_block = pred_image[tl[0]:br[0], tl[1]:br[1]]
            else:
                input_block = pred_image[tl[0]:br[0], tl[1]:br[1], :]
            self._process_block(input_block, sx + tl[0], sy + tl[1],
                                None if labels is None else labels[tl[0]:br[0], tl[1]:br[1]], label_nodata)

        try:
            image.process_rois(tiles, callback_function, show_progress=self._show_progress)
        except KeyboardInterrupt:
            self._abort()
            raise

        return self._complete()

class LabelPredictor(Predictor):
    """
    Predicts integer labels for an image.
    """
    def __init__(self, model, tile_shape=None, output_image=None, show_progress=False, # pylint:disable=too-many-arguments
                 colormap=None, prob_image=None, error_image=None, error_colors=None):
        """
        Parameters
        ----------
        model: tensorflow.keras.models.Model
            Model to evaluate.
        tile_shape: (int, int)
            Shape of tiles to process.
        output_image: str
            If specified, output the results to this image.
        show_progress: bool
            Print progress to command line.
        colormap: List[Any]
            Map classes to colors given in the colormap.
        prob_image: str
            If given, output a probability image to this file. Probabilities are scaled as bytes
            1-255, with 0 as nodata.
        error_image: str
            If given, output an image showing where the classification is incorrect.
        error_colors: List[Any]
            Colormap for the error_image.
        """
        super().__init__(model, tile_shape, show_progress)
        self._confusion_matrix = None
        self._num_classes = None
        self._output_image = output_image
        if colormap is not None:
            # convert python list to numpy array
            if not isinstance(colormap, np.ndarray):
                a = np.zeros(shape=(len(colormap), 3), dtype=np.uint8)
                for (i, v) in enumerate(colormap):
                    a[i][0] = (v >> 16) & 0xFF
                    a[i][1] = (v >> 8) & 0xFF
                    a[i][2] = v & 0xFF
                colormap = a
        self._colormap = colormap
        self._prob_image = prob_image
        self._error_image = error_image
        self._error_colors = error_colors
        if self._error_image:
            assert self._error_colors is not None, 'Must specify error_colors.'
        self._output = None
        self._prob_o = None
        self._errors = None

    def _initialize(self, shape, image, label=None):
        net_output_shape = self._model.output_shape[1:]
        self._num_classes = net_output_shape[-1]
        if self._prob_image:
            self._prob_image.initialize((shape[0], shape[1], self._num_classes), np.dtype(np.uint8),
                                        image.metadata(), nodata_value=0)

        if self._num_classes == 1: # special case
            self._num_classes = 2
        if self._colormap is not None and self._num_classes != self._colormap.shape[0]:
            print('Warning: Defined defined classes (%d) in config do not match network (%d).' %
                  (self._colormap.shape[0], self._num_classes))
            if self._colormap.shape[0] > self._num_classes:
                self._num_classes = self._colormap.shape[0]

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
                self._output_image.initialize((shape[0], shape[1]), np.int32, image.metadata())
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

    def _process_block(self, pred_image, x, y, labels, label_nodata):
        if self._prob_image is not None:
            prob = 1.0 + (pred_image * 254.0)
            prob = prob.astype(np.uint8)
            prob[np.isnan(pred_image[:, :, 0] if len(pred_image.shape) == 3 else pred_image)] = 0
            self._prob_image.write(prob, x, y)

        if labels is None and self._output_image is None:
            return

        prob_image = pred_image
        if len(pred_image.shape) == 2:
            pred_image[~np.isnan(pred_image)] = pred_image[~np.isnan(pred_image)] >= 0.5
            pred_image = pred_image.astype(int)
            prob_image = np.expand_dims(prob_image, -1)
        else:
            pred_image = np.argmax(pred_image, axis=2)

        # nodata pixels were set to nan in the probability image
        pred_image[np.isnan(prob_image[:, :, 0])] = -1

        if labels is not None:
            incorrect = (labels != pred_image).astype(int)

            valid_labels = labels
            valid_pred = pred_image
            if label_nodata is not None:
                invalid = np.logical_or((labels == label_nodata), pred_image == -1)
                valid = np.logical_not(invalid)
                incorrect[invalid] = 0
                valid_labels = labels[valid]
                valid_pred = pred_image[valid]

            if self._error_image:
                self._error_image.write(self._error_colors[incorrect], x, y)
            cm = tf.math.confusion_matrix(np.ndarray.flatten(valid_labels),
                                          np.ndarray.flatten(valid_pred),
                                          self._num_classes)
            self._confusion_matrix[:, :] += cm

        if self._output_image is not None:
            if self._colormap is not None:
                colormap = np.zeros((self._colormap.shape[0] + 1, self._colormap.shape[1]))
                colormap[0:-1, :] = self._colormap
                if labels is not None and label_nodata is not None:
                    pred_image[pred_image == -1] = self._colormap.shape[0]
                result = np.zeros((pred_image.shape[0], pred_image.shape[1], self._colormap.shape[1]))
                for i in range(prob_image.shape[2]):
                    result += (colormap[i, :] * prob_image[:, :, i, np.newaxis]).astype(colormap.dtype)
                self._output_image.write(result, x, y)
            else:
                self._output_image.write(pred_image, x, y)

    def confusion_matrix(self):
        """
        Returns a matrix counting true labels matched to predicted labels.
        """
        return self._confusion_matrix

class ImagePredictor(Predictor):
    """
    Predicts an image from an image.
    """
    def __init__(self, model, tile_shape=None, output_image=None, show_progress=False, transform=None):
        """
        Parameters
        ----------
        model: tensorflow.keras.models.Model
            Model to evaluate.
        tile_shape: (int, int)
            Shape of tiles to process at a time.
        output_image: str
            File to output results to.
        show_progress: bool
            Print progress to screen.
        transform: (Callable[[numpy.ndarray], numpy.ndarray], output_type, num_bands)
            The callable will be applied to the results from the network before saving
            to a file. The results should be of type output_type and the third dimension
            should be size num_bands.
        """
        super().__init__(model, tile_shape, show_progress)
        self._output_image = output_image
        self._output = None
        self._transform = transform

    def _initialize(self, shape, image, label=None):
        net_output_shape = self._model.output_shape[1:]
        if self._output_image is not None:
            dtype = np.float32 if self._transform is None else np.dtype(self._transform[1])
            bands = net_output_shape[-1] if self._transform is None else self._transform[2]
            self._output_image.initialize((shape[0], shape[1], bands), dtype, image.metadata())

    def _complete(self):
        if self._output_image is not None:
            self._output_image.close()

    def _abort(self):
        self._complete()
        if self._output_image is not None:
            self._output_image.abort()

    def _process_block(self, pred_image, x, y, labels, label_nodata):
        if self._output_image is not None:
            im = pred_image
            if self._transform is not None:
                im = self._transform[0](im)
            self._output_image.write(im, x, y)
