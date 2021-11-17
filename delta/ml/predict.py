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
import PIL
from PIL import ImageDraw #pylint: disable=W0611
from numpy.core.fromnumeric import take #pylint: disable=W0611
import tensorflow as tf

from delta.imagery.rectangle import Rectangle

def mask_outside_shapes(shapes, image, y, x, mask_value):
    '''Writes the mask value to "image" in all locations outside one of the shapes.
       y,x = the top left coordinate of "image" in the full uncropped image'''

    if not shapes: # Skip this step if no shapes were passed in
        return

    image_rect = Rectangle(x, y, width=image.shape[1], height=image.shape[0])

    def adjust_coords(inputs, x, y):
        '''Return a copy of the input coordinates adjusted by the x,y coordinate'''
        output = []
        for pair in inputs:
            output.append((pair[0] - x, pair[1] - y))
        return output

    # Draw all of the shapes on to a new image
    mask_image = None
    painter = None
    for s in shapes:
        shape_rect = Rectangle(*s.bounds)
        if not shape_rect.overlaps(image_rect):
            continue
        if not mask_image:
            mask_image = PIL.Image.new('L', (image.shape[1], image.shape[0]), color=0)
            painter = PIL.ImageDraw.Draw(mask_image)
        coords = adjust_coords(s.exterior.coords, x, y)
        painter.polygon(coords, fill=1)
        for i in s.interiors: # Interior polygons are "holes" in the shapes
            coords = adjust_coords(i.coords, x, y)
            painter.polygon(coords, fill=0)
    if not mask_image:
        return

    # Apply the mask to the input image
    mask_pixels = mask_image.load()
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if not mask_pixels[c,r]:
                image[r,c,0] = mask_value

class Predictor(ABC):
    """
    Abstract class to run prediction for an image given a model.
    """
    def __init__(self, model, tile_shape=None, show_progress=False, progress_text=None):
        self._model = model
        self._show_progress = show_progress
        self._progress_text = progress_text
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
    def _process_block(self, pred_image: np.ndarray, y: int, x: int, labels: np.ndarray, label_nodata):
        """
        Processes a predicted block. Must be overriden in subclasses.

        Parameters
        ----------
        pred_image: numpy.ndarray
            Output of model for a block of the image.
        y: int
            Top-left y coordinate of block.
        x: int
            Top-left x coordinate of block.
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
            if not result.flags['WRITEABLE']: # older tensorflow version
                result = np.array(result)
            if image_nodata_value is not None:
                x0 = (data.shape[1] - result.shape[1]) // 2
                y0 = (data.shape[0] - result.shape[0]) // 2
                invalid = (data if len(data.shape) == 2 else \
                          data[:, :, 0])[y0:y0 + result.shape[0], x0:x0 + result.shape[1]] == image_nodata_value
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
            ox = (data.shape[1] - out_shape[1]) // 2
            oy = (data.shape[0] - out_shape[0]) // 2
            output_slice = data[oy:-oy, ox:-ox, 0]
            retval[output_slice == image_nodata_value] = math.nan

        return retval

    def predict(self, image, label=None, input_bounds=None, overlap=(0, 0),
                roi_shapes=None):
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
            by the given amounts in the y and x directions. Subclasses may select or interpolate
            to favor tile interior pixels for improved classification.
        roi_shapes: List[shapely.geometry.BaseGeometry]
            A list of the shapes that we want to compute results for.  All values are in pixel
            coordinates.

        Returns
        -------
        The result of the `_complete` function, which depends on the sublcass.
        """
        net_input_shape = self._model.input_shape[1:]
        net_output_shape = self._model.output_shape[1:]

        # Set up the output image
        image_rect = Rectangle(0, 0, width=image.width(), height=image.height())
        if not input_bounds:
            input_bounds = image_rect
        output_shape = (input_bounds.height(), input_bounds.width())

        ts = self._tile_shape if self._tile_shape else (image.height(), image.width())

        if net_input_shape[0] is None and net_input_shape[1] is None:
            assert net_output_shape[0] is None and net_output_shape[1] is None
            out_shape = self._model.compute_output_shape((0, ts[0], ts[1], net_input_shape[2]))

            overlap_shape=(ts[0] - out_shape[1] + overlap[0],
                           ts[1] - out_shape[2] + overlap[1])
            tiles, tiles_valid = input_bounds.make_tile_rois_yx(ts, include_partials=False,
                                                                overlap_shape=overlap_shape,
                                                                partials_overlap=True, containing_rect=image_rect)

        else:
            offset_r = -net_input_shape[0] + net_output_shape[0] + overlap[0]
            offset_c = -net_input_shape[1] + net_output_shape[1] + overlap[1]
            overlap_shape=(net_input_shape[0] - net_output_shape[0] + overlap[0],
                           net_input_shape[1] - net_output_shape[1] + overlap[1])
            output_shape = (output_shape[0] + offset_r, output_shape[1] + offset_c)

            tiles, tiles_valid = input_bounds.make_tile_rois_yx((net_input_shape[0], net_input_shape[1]),
                                                                include_partials=False,
                                                                overlap_shape=overlap_shape,
                                                                partials_overlap=True, containing_rect=image_rect)

        self._initialize(output_shape, image, label)

        label_nodata = label.nodata_value() if label else None

        def callback_function(roi, data, tile_valid_roi):
            ''''''
            # Pred_image should correspond to 'roi'
            pred_image = self._predict_array(data, image.nodata_value())

            # Next we crop pred_image down to the valid (non_overlapping) output area,
            # this coordinate is the top left location of the cropped image in the final output
            # image that we will write.
            relative_x = tile_valid_roi.min_x - input_bounds.min_x
            relative_y = tile_valid_roi.min_y - input_bounds.min_y

            # Convert the valid ROI to be relative to this individual tile ROI
            local_valid_roi = Rectangle(tile_valid_roi.min_x - roi.min_x,
                                        tile_valid_roi.min_y - roi.min_y,
                                        width=tile_valid_roi.width(),
                                        height=tile_valid_roi.height())

            labels = None
            if label:
                # Read the corresponding portion of the label image
                cropped_label = label.read(tile_valid_roi)
                # Mask all regions outside the specified shapes
                mask_outside_shapes(roi_shapes, cropped_label, tile_valid_roi.min_y, tile_valid_roi.min_x, label_nodata)
                labels = np.squeeze(cropped_label)

            # Crop out the valid portion of the image
            if len(pred_image.shape) == 2:
                input_block = pred_image[local_valid_roi.min_y:local_valid_roi.max_y,
                                         local_valid_roi.min_x:local_valid_roi.max_x]
            else:
                input_block = pred_image[local_valid_roi.min_y:local_valid_roi.max_y,
                                         local_valid_roi.min_x:local_valid_roi.max_x, :]

            self._process_block(input_block, relative_y, relative_x,
                                labels, label_nodata)

        try:
            image.process_rois(tiles, callback_function, show_progress=self._show_progress,
                               progress_prefix=self._progress_text, roi_extra_data=tiles_valid)
        except KeyboardInterrupt:
            self._abort()
            raise

        return self._complete()

class LabelPredictor(Predictor):
    """
    Predicts integer labels for an image.
    """
    def __init__(self, model, tile_shape=None, output_image=None, show_progress=False, progress_text=None, # pylint:disable=too-many-arguments
                 colormap=None, prob_image=None, error_image=None, error_abs=False, metrics=None):
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
            Map classes to colors given in the colormap for output_image.
        prob_image: str
            If given, output a probability image to this file. Probabilities are scaled as bytes
            1-255, with 0 as nodata.
        error_image: delta.extensions.sources.tiff.TiffWriter
            If given, outputs an error image showing the difference between the predicted probability and
            the binary label. i.e. prediction - label. The values [-1,1] are linearly scaled and clipped as bytes
            [1-255], with 0 as nodata.
        error_abs: bool
            If True, the error_image will be the absolute value of (prediction - label). i.e. abs(prediction - label).
            The values [0,1] are linearly scaled and clipped as bytes [1-255], with 0 as nodata.
        error_colors: List[Any]
            Colormap for the error_image.
        metrics: List[Any]
            List of Metric class instances to compute with (requires labels).
        """
        super().__init__(model, tile_shape, show_progress, progress_text)
        self._confusion_matrix = None
        self._num_classes = None
        self._output_image = output_image
        if output_image is None:
            colormap = None
        elif colormap is not None:
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
        self._error_abs = error_abs
        self._output = None
        self._prob_o = None
        self._metrics = metrics

    def _initialize(self, shape, image, label=None):
        net_output_shape = self._model.output_shape[1:]
        self._num_classes = net_output_shape[-1]
        if self._prob_image:
            self._prob_image.initialize((shape[0], shape[1], self._num_classes), np.dtype(np.uint8),
                                        image.metadata(), nodata_value=0)

        if self._error_image:
            self._error_image.initialize((shape[0], shape[1], self._num_classes), np.dtype(np.uint8), image.metadata(),
                                         nodata_value=0)

        if self._num_classes == 1: # special case
            self._num_classes = 2
        if self._colormap is not None and self._num_classes != self._colormap.shape[0]:
            print('Warning: Defined defined classes (%d) in config do not match network (%d).' %
                  (self._colormap.shape[0], self._num_classes))
            if self._colormap.shape[0] > self._num_classes:
                self._num_classes = self._colormap.shape[0]
        if label:
            self._confusion_matrix = np.zeros((self._num_classes, self._num_classes), dtype=np.int32)
        else:
            self._confusion_matrix = None
        if self._output_image:
            if self._colormap is not None:
                self._output_image.initialize((shape[0], shape[1], self._colormap.shape[1]),
                                              self._colormap.dtype, image.metadata())
            else:
                self._output_image.initialize((shape[0], shape[1]), np.int32, image.metadata())

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

    def _process_block(self, pred_image, y, x, labels, label_nodata):

        # create a masked array. The mask is true where pred_image = np.nan
        if len(pred_image.shape) == 3 and pred_image.shape[2] == 1:
            pred_image = np.squeeze(pred_image, axis=2)
        pred_image_ma = np.ma.masked_invalid(pred_image)
        if len(pred_image_ma.shape) == 3:
            # sets the first layer mask as the mask for all layers
            pred_first_layer_mask_duplicated = \
                np.repeat(pred_image_ma.mask[:,:,0,np.newaxis], repeats=pred_image_ma.shape[2], axis=2)
            pred_image_ma.mask = pred_first_layer_mask_duplicated

        labels_ma = np.ma.masked_equal(labels, label_nodata)

        if self._prob_image is not None:
            # scale and clip image to 1-255 with 0 being reserved for nodata where pred_image is nan
            prob = np.clip((pred_image_ma * 254.0).astype(np.uint8), 0, 254)
            prob = 1 + prob
            # fill nodata values in array with 0
            prob = prob.filled(0)
            self._prob_image.write(prob, y, x)

        if labels is None and self._output_image is None:
            return

        if len(pred_image_ma.shape) == 2:
            # convert prediction image from continuous to binary: 1 where => 0.5 and 0 where < 0.5
            class_int_image = (pred_image_ma >= 0.5).astype(int)
        else:
            # returns the indicies of the maximum value bound by the axis
            # this would return 0-max depth depending on max value. Selects which class prediction was highest
            class_int_image = np.ma.MaskedArray(np.argmax(pred_image_ma, axis=2), mask=pred_image_ma.mask[:,:,0])

        if labels is not None:
            # identify incorrect predictions
            incorrect = (labels_ma != class_int_image).astype(int)

            # combine the masks for labels and pred_image
            # you can't have a valid label where prediction is invalid and vice versa
            valid_labels = labels_ma.copy()
            valid_labels.mask = incorrect.mask
            valid_pred_class = class_int_image.copy()
            valid_pred_class.mask = incorrect.mask

            if self._error_image:
                # TODO: implement for multiclass prediction
                # will need to separate out labels into different labels so that errors can have their own image label
                if len(pred_image_ma.shape) == 3:
                    raise NotImplementedError

                continuous_error = pred_image_ma - labels_ma

                if self._error_abs:
                    continuous_abs_error = np.abs(continuous_error)
                    # shift and int continuous error for image output
                    continuous_abs_error_inted = np.clip(((continuous_abs_error * 254) + 1).astype(np.uint8), 1, 255)
                    # fill nodata values in array with 0 and write to image
                    self._error_image.write(continuous_abs_error_inted.filled(0), y, x)
                else:
                    # shift and int continuous error for image output
                    continuous_error_inted = np.clip(((continuous_error * 127) + 128).astype(np.uint8), 1, 255)
                    # fill nodata values in array with 0 and write to image
                    self._error_image.write(continuous_error_inted.filled(0), y, x)

            vlcomp = valid_labels.compressed() # pylint: disable=no-member
            cm = tf.math.confusion_matrix(vlcomp, valid_pred_class.compressed(), self._num_classes)
            self._confusion_matrix[:, :] += cm

            if self._metrics and vlcomp.size > 0:
                valid_pred = pred_image_ma.copy()
                valid_pred.mask = incorrect.mask
                vpcomp = valid_pred.compressed() #pylint: disable=no-member
                for m in self._metrics:
                    m.update_state(vlcomp, vpcomp)

        self._save_output_image(pred_image_ma, class_int_image, y, x)

    def _save_output_image(self, pred_image_ma, class_int_image, y, x):
        if self._output_image is None:
            return
        if self._colormap is not None:
            # create a third entry in the color map that is all 0s
            colormap = np.zeros((self._colormap.shape[0] + 1, self._colormap.shape[1]))
            colormap[0:-1, :] = self._colormap

            # create array to be filled with correct shape
            result = np.zeros((pred_image_ma.shape[0], pred_image_ma.shape[1], self._colormap.shape[1]))

            # When pred_image.shape is 2, then prob_image is the binarized version of the paramater
            # pred_image passed to function. When pred_image.shape is 3,
            # prob_image is just the original pred_image passed to the function.
            if len(pred_image_ma.shape) == 2:
                result_image = np.expand_dims(class_int_image, -1)
            else:
                result_image = pred_image_ma

            # for each layer of prob_image
            for i in range(result_image.shape[2]):
                # assign the appropriate color map value for each layer. This will result in a single layer
                # with the appropriate colors added for each layer if an inted version of the image is used.
                # When just water is predicted this results in a single color. However when multiple types
                # are predicted this will result in a probability blended mixture of colors.
                result += (colormap[i, :] * result_image[:, :, i, np.newaxis]).filled(np.nan).astype(colormap.dtype)

            # result is written as output image and nans aren't filled with anything. Just numpy float nan
            self._output_image.write(result, y, x)
        else:
            # fill nodata values in array with -1 and write to output image
            self._output_image.write(class_int_image.filled(-1), y, x)

    def confusion_matrix(self):
        """
        Returns a matrix counting true labels matched to predicted labels.
        """
        return self._confusion_matrix

    def metrics(self):
        """
        Returns the list of metric objects
        """
        if self._metrics is None:
            return []
        return self._metrics


class ImagePredictor(Predictor):
    """
    Predicts an image from an image.
    """
    def __init__(self, model, tile_shape=None, output_image=None, show_progress=False,
                 progress_text=None, transform=None):
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
        super().__init__(model, tile_shape, show_progress, progress_text)
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

    def _process_block(self, pred_image, y, x, labels, label_nodata):
        if self._output_image is not None:
            im = pred_image
            if self._transform is not None:
                im = self._transform[0](im)
            self._output_image.write(im, y, x)
