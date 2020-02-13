#pylint: disable=redefined-outer-name
import os

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from delta.config import config, TrainingSpec
from delta.imagery import imagery_dataset
from delta.imagery.sources import tfrecord, npy
from delta.ml import train, predict

import conftest

def load_dataset(source, output_size):
    config.reset() # don't load any user files
    (image_path, label_path) = source[0]
    config.load(yaml_str=
                '''
                general:
                  cache:
                    dir: %s
                images:
                  type: %s
                  directory: %s
                  extension: %s
                  preprocess: False
                labels:
                  type: %s
                  directory: %s
                  extension: %s
                  preprocess: False
                network:
                  chunk_size: 3
                mlflow:
                  enabled: false''' %
                (os.path.dirname(image_path), source[2], os.path.dirname(image_path), source[1],
                 source[4], os.path.dirname(label_path), source[3]))

    dataset = imagery_dataset.ImageryDataset(config.images(), config.labels(),
                                             config.chunk_size(), output_size,
                                             config.training().chunk_stride)
    return dataset

@pytest.fixture(scope="function", params=range(2))
def dataset(all_sources, request):
    source = all_sources[request.param]
    return load_dataset(source, 1)

@pytest.fixture(scope="function")
def dataset_block_label(all_sources):
    return load_dataset(all_sources[0], 3)

def test_tfrecord_write(tfrecord_filenames):
    """
    Write and read from disks, but only reading full images, not chunked data
    from ImageryDataset.
    """
    images = tfrecord.create_dataset([tfrecord_filenames[0]], 1, tf.float32)
    labels = tfrecord.create_dataset([tfrecord_filenames[1]], 1, tf.uint8)
    ds = tf.data.Dataset.zip((images, labels))
    for value in ds.take(100):
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
        unique = np.unique(img)
        assert (0 in unique or 1 in unique and len(unique) <= 2)
        num_data += 1
    num_label = 0
    for label in dataset.labels():
        num_label += 1
    assert num_label == num_data

    ds = dataset.dataset()
    for (image, label) in ds.take(100):
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

def test_block_label(dataset_block_label): #pylint: disable=redefined-outer-name
    """
    Same as previous test but with dataset that gives labels as 3x3 blocks.
    """
    num_data = 0
    for image in dataset_block_label.data():
        img = image.numpy()
        assert img.dtype == np.float32
        unique = np.unique(img)
        assert (0 in unique or 1 in unique and len(unique) <= 2)
        num_data += 1
    num_label = 0
    for label in dataset_block_label.labels():
        num_label += 1
    assert num_label == num_data

    ds = dataset_block_label.dataset()
    for (image, label) in ds.take(100):
        if label[1, 1]:
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
            assert label[1, 1] == 0
        v4 = image[1][0][0] == 0
        v5 = image[1][2][0] == 0
        if v4 or v5:
            assert label[1, 1] == 0
        v6 = image[2][0][0] == 0
        v7 = image[2][1][0] == 0
        v8 = image[2][2][0] == 0
        if v6 or v7 or v8:
            assert label[1, 1] == 0

def test_train(dataset): #pylint: disable=redefined-outer-name
    def model_fn():
        return keras.Sequential([
            keras.layers.Flatten(input_shape=(3, 3, 1)),
            keras.layers.Dense(3 * 3, activation=tf.nn.relu),
            keras.layers.Dense(2, activation=tf.nn.softmax),
            keras.layers.Reshape((1, 1, 2))
            ])
    model, _ = train.train(model_fn, dataset,
                           TrainingSpec(100, 5, 'sparse_categorical_crossentropy'))
    ret = model.evaluate(x=dataset.dataset().batch(1000))
    assert ret[1] > 0.90

    (test_image, test_label) = conftest.generate_tile()
    test_label = test_label[1:-1, 1:-1]
    result = predict.predict(model, npy.NumpyImage(test_image))
    assert sum(sum(np.logical_xor(result, test_label))) < 200 # very easy test since we don't train much

@pytest.fixture(scope="function")
def autoencoder(all_sources):
    source = all_sources[0]
    config.reset() # don't load any user files
    (image_path, _) = source[0]
    config.load(yaml_str=
                '''
                general:
                  cache:
                    dir: %s
                images:
                  type: %s
                  directory: %s
                  extension: %s
                  preprocess: False
                network:
                  chunk_size: 3
                mlflow:
                  enabled: false''' %
                (os.path.dirname(image_path), source[2], os.path.dirname(image_path), source[1]))

    dataset = imagery_dataset.AutoencoderDataset(config.images(),
                                                 config.chunk_size(), config.training().chunk_stride)
    return dataset

def test_autoencoder(autoencoder): #pylint: disable=redefined-outer-name
    """
    Test that the inputs and outputs of the dataset are the same.
    """
    ds = autoencoder.dataset()
    for (image, label) in ds.take(1000):
        assert (image.numpy() == label.numpy()).all()
