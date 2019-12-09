#pylint: disable=redefined-outer-name
import os

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from delta.config import config
from delta.imagery import imagery_dataset
from delta.imagery.sources import tfrecord
from delta.ml import train

@pytest.fixture(scope="function", params=range(2))
def dataset(all_sources, request):
    source = all_sources[request.param]
    config.reset() # don't load any user files
    (image_path, label_path) = source[0]
    config.set_value('input_dataset', 'extension', source[1])
    config.set_value('input_dataset', 'image_type', source[2])
    config.set_value('input_dataset', 'file_type', source[3])
    config.set_value('input_dataset', 'label_extension', source[4])
    config.set_value('input_dataset', 'label_type', source[5])
    config.set_value('input_dataset', 'data_directory', os.path.dirname(image_path))
    config.set_value('input_dataset', 'label_directory', os.path.dirname(label_path))
    config.set_value('ml', 'chunk_size', 3)
    config.set_value('cache', 'cache_dir', os.path.dirname(image_path))
    dataset = imagery_dataset.ImageryDataset(config.dataset(), config.chunk_size(), config.chunk_stride())
    return dataset

def test_tfrecord_write(tfrecord_filenames):
    """
    Write and read from disks, but only reading full images, not chunked data
    from ImageryDataset.
    """
    images = tfrecord.create_dataset([tfrecord_filenames[0]], 1, tf.float32)
    labels = tfrecord.create_dataset([tfrecord_filenames[1]], 1, tf.uint8)
    ds = tf.data.Dataset.zip((images, labels))
    for value in ds:
        image = tf.squeeze(value[0])
        label = tf.squeeze(value[1])
        assert image.shape == label.shape
        for x in range(1, image.shape[0] - 1):
            for y in range(1, image.shape[1] - 1):
                if label[x][y]:
                    assert image[x-1][y-1] == 1
                    assert image[x-1][y  ] == 1
                    assert image[x-1][y+1] == 1
                    assert image[x  ][y-1] == 1
                    assert image[x  ][y+1] == 1
                    assert image[x+1][y-1] == 1
                    assert image[x+1][y  ] == 1
                    assert image[x+1][y+1] == 1
                assert image[x][y] == 1 or image[x][y] == 0

def test_tfrecord_write_read(dataset): #pylint: disable=redefined-outer-name
    """
    Writes and reads from disks, then checks if what is read is valid according to the
    generation procedure.
    """
    num_data = 0
    for image in dataset.data():
        img = image.numpy()
        assert img.dtype == np.float32
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                assert img[i][j][0] == 1 or img[i][j][0] == 0
        num_data += 1
    num_label = 0
    for label in dataset.labels():
        num_label += 1
    assert num_label == num_data

    ds = dataset.dataset(filter_zero=False)
    for (image, label) in ds:
        try:
            if label:
                assert image[0][0][0] == 1
                assert image[0][1][0] == 1
                assert image[0][2][0] == 1
                assert image[1][0][0] == 1
                assert image[1][2][0] == 1
                assert image[2][0][0] == 1
                assert image[2][1][0] == 1
                assert image[2][2][0] == 1
            v1 = image[0][0][0] == 0
            v2 = image[0][1][0] == 0
            v3 = image[0][2][0] == 0
            if v1 or v2 or v3:
                assert label == 0
            v4 = image[1][0][0] == 0
            v5 = image[1][2][0] == 0
            if v4 or v5:
                assert label == 0
            v6 = image[2][0][0] == 0
            v7 = image[2][1][0] == 0
            v8 = image[2][2][0] == 0
            if v6 or v7 or v8:
                assert label == 0
        except tf.errors.OutOfRangeError:
            break


def test_train(dataset): #pylint: disable=redefined-outer-name
    def model_fn():
        return  keras.Sequential([
            keras.layers.Conv2D(1, kernel_size=(3, 3),
                                kernel_initializer=keras.initializers.Zeros(),
                                activation='relu', data_format='channels_last',
                                input_shape=(3, 3, 1))])
    def create_dataset():
        d = dataset.dataset(filter_zero=False)
        d = d.batch(100).repeat(5)
        return d

    # Ignoring returned model training history to keep pylint happy.
    model, _ = train.train(model_fn, create_dataset(),
                           optimizer=tf.optimizers.Adam(learning_rate=0.01),
                           loss_fn='mean_squared_logarithmic_error',
                           validation_data=None,
                           num_gpus=0)
    ret = model.evaluate(x=create_dataset())
    assert ret[1] > 0.90
