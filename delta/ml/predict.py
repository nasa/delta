import numpy as np
from numpy.lib.stride_tricks import as_strided

from delta.imagery import rectangle

def predict_array(model, cs, data):
    out_shape = (data.shape[0] - cs + 1, data.shape[1] - cs + 1)
    chunks = as_strided(data, shape=(out_shape[0], out_shape[1], cs, cs, data.shape[2]),
                        strides=(data.strides[0], data.strides[1], data.strides[0],
                                 data.strides[1], data.strides[2]),
                        writeable=False)
    chunks = np.reshape(chunks, (-1, cs, cs, data.shape[2]))
    predictions = model.predict_on_batch(chunks)
    best = np.argmax(predictions, axis=1)
    return np.reshape(best, (out_shape[0], out_shape[1]))

def predict_validate(model, cs, image, label, input_bounds=None, show_progress=False):
    """Like predict but returns (predicted image, error image, percent correct)."""
    block_size_x = 256
    block_size_y = 256

    # Set up the output image
    if not input_bounds:
        input_bounds = rectangle.Rectangle(0, 0, width=image.width(), height=image.height())

    result = np.zeros((input_bounds.width() - cs + 1, input_bounds.height() - cs + 1), dtype=np.uint8)
    errors = None
    if label:
        errors = np.zeros((input_bounds.width() - cs + 1, input_bounds.height() - cs + 1), dtype=np.bool)

    num_wrong = []
    def callback_function(roi, data):
        image = predict_array(model, cs, data)

        block_x = (roi.min_x - input_bounds.min_x) // block_size_x
        block_y = (roi.min_y - input_bounds.min_y) // block_size_y
        (sx, sy) = (block_x * block_size_x, block_y * block_size_y)
        result[sx : sx + image.shape[0], sy : sy + image.shape[1]] = image
        if label:
            label_roi = rectangle.Rectangle(roi.min_x + (cs // 2), roi.min_y + (cs // 2),
                                            roi.max_x - (cs // 2), roi.max_y - (cs // 2))
            wrong = np.squeeze(label.read(label_roi)) != image
            errors[sx : sx + image.shape[0], sy : sy + image.shape[1]] = wrong
            num_wrong.append(np.sum(wrong))

    output_rois = input_bounds.make_tile_rois(block_size_x + cs - 1, block_size_y + cs - 1,
                                              include_partials=True, overlap_amount=cs - 1)

    image.process_rois(output_rois, callback_function, show_progress=show_progress)
    return (result, errors, 1.0 - sum(num_wrong) / (result.shape[0] * result.shape[1]))#pylint:disable=unsubscriptable-object

def predict(model, cs, image, input_bounds=None, show_progress=False):
    """Returns the predicted image given a model, chunk size, and image."""
    return predict_validate(model, cs, image, None, input_bounds, show_progress)[0]
