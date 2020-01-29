import numpy as np

import tensorflow as tf

from delta.imagery import rectangle

def predict_array(model, cs, data):
    out_shape = (data.shape[0] - cs + 1, data.shape[1] - cs + 1)
    image = tf.convert_to_tensor(data)
    image = tf.expand_dims(image, 0)
    chunks = tf.image.extract_patches(image, [1, cs, cs, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding='VALID')
    chunks = tf.reshape(chunks, [-1, cs, cs, data.shape[2]])
    best = np.zeros((chunks.shape[0],), dtype=np.int32)
    ## TODO: other data types, configurable batch size
    BATCH_SIZE=1000
    for i in range(0, chunks.shape[0], BATCH_SIZE):
        best[i:i+BATCH_SIZE] = np.argmax(model.predict_on_batch(chunks[i:i+BATCH_SIZE]), axis=1)
    return np.reshape(best, (out_shape[0], out_shape[1]))

def predict_validate(model, cs, image, label, num_classes, input_bounds=None, show_progress=False):
    """Like predict but returns (predicted image, error image, percent correct)."""
    block_size_x = 256
    block_size_y = 256

    # Set up the output image
    if not input_bounds:
        input_bounds = rectangle.Rectangle(0, 0, width=image.width(), height=image.height())

    result = np.zeros((input_bounds.width() - cs + 1, input_bounds.height() - cs + 1), dtype=np.uint8)
    errors = None
    confusion_matrix = None
    if label:
        errors = np.zeros((input_bounds.width() - cs + 1, input_bounds.height() - cs + 1), dtype=np.bool)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    def callback_function(roi, data):
        image = predict_array(model, cs, data)

        block_x = (roi.min_x - input_bounds.min_x) // block_size_x
        block_y = (roi.min_y - input_bounds.min_y) // block_size_y
        (sx, sy) = (block_x * block_size_x, block_y * block_size_y)
        result[sx : sx + image.shape[0], sy : sy + image.shape[1]] = image
        if label:
            label_roi = rectangle.Rectangle(roi.min_x + (cs // 2), roi.min_y + (cs // 2),
                                            roi.max_x - (cs // 2), roi.max_y - (cs // 2))
            labels = np.squeeze(label.read(label_roi))
            errors[sx : sx + image.shape[0], sy : sy + image.shape[1]] = labels != image
            cm = tf.math.confusion_matrix(np.ndarray.flatten(labels), np.ndarray.flatten(image), num_classes)
            confusion_matrix[:, :] += cm

    output_rois = input_bounds.make_tile_rois(block_size_x + cs - 1, block_size_y + cs - 1,
                                              include_partials=True, overlap_amount=cs - 1)

    image.process_rois(output_rois, callback_function, show_progress=show_progress)
    return (result, errors, confusion_matrix)

def predict(model, cs, image, num_classes, input_bounds=None, show_progress=False):
    """Returns the predicted image given a model, chunk size, and image."""
    return predict_validate(model, cs, image, None, num_classes, input_bounds, show_progress)[0]
