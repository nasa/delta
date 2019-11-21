#pylint: disable=redefined-outer-name
import pytest
import numpy as np
import tensorflow as tf

from delta.imagery.sources import tfrecord, tiff

@pytest.fixture(scope="function")
def tiff_image(tmp_path):
    width = 64
    height = 32
    numpy_image = np.random.rand(width, height, 3).astype(np.float32)
    filename = str(tmp_path / 'test.tiff')
    tiff.write_tiff(filename, numpy_image)
    return (numpy_image, filename)

def test_save_tfrecord(tiff_image, tmp_path):
    filename = str(tmp_path / 'test.tfrecord')
    image = tiff.TiffImage(tiff_image[1])
    tfrecord.image_to_tfrecord(image, [filename], show_progress=False)
    ds = tfrecord.create_dataset([filename], image.num_bands(), tf.float32)
    for value in ds:
        value = np.squeeze(value.numpy())
        assert value.shape == tiff_image[0].shape
        assert np.allclose(value, tiff_image[0])

def test_save_tiles(tiff_image, tmp_path):
    filename = str(tmp_path / 'test.tfrecord')
    image = tiff.TiffImage(tiff_image[1])
    tfrecord.image_to_tfrecord(image, [filename], (30, 30), show_progress=False)
    ds = tfrecord.create_dataset([filename], image.num_bands(), tf.float32)
    area = 0
    for value in ds:
        value = np.squeeze(value.numpy())
        assert value.shape[0] == 30 or value.shape[0] == 4
        assert value.shape[1] == 30 or value.shape[1] == 2
        area += value.shape[0] * value.shape[1]
    assert area == 64 * 32

def test_save_tiles_no_partial(tiff_image, tmp_path):
    filename = str(tmp_path / 'test.tfrecord')
    image = tiff.TiffImage(tiff_image[1])
    tfrecord.image_to_tfrecord(image, [filename], (30, 30), include_partials=False, show_progress=False)
    ds = tfrecord.create_dataset([filename], image.num_bands(), tf.float32)
    area = 0
    for value in ds:
        value = np.squeeze(value.numpy())
        assert value.shape[0] == 30
        assert value.shape[1] == 30
        area += value.shape[0] * value.shape[1]
    assert area == 60 * 30

def test_save_overlap(tiff_image, tmp_path):
    filename = str(tmp_path / 'test.tfrecord')
    image = tiff.TiffImage(tiff_image[1])
    tfrecord.image_to_tfrecord(image, [filename], (30, 30), overlap_amount=5, show_progress=False)
    ds = tfrecord.create_dataset([filename], image.num_bands(), tf.float32)
    area = 0
    for value in ds:
        value = np.squeeze(value.numpy())
        assert value.shape[0] == 30 or value.shape[0] == 14
        assert value.shape[1] == 30 or value.shape[1] == 7
        area += value.shape[0] * value.shape[1]
    assert area == 30 * 30 * 2 + 30 * 7 * 2 + 14 * 30 * 1 + 14 * 7 * 1

def test_save_single_band(tiff_image, tmp_path):
    filename = str(tmp_path / 'test.tfrecord')
    image = tiff.TiffImage(tiff_image[1])
    tfrecord.image_to_tfrecord(image, [filename], bands_to_use=[2], show_progress=False)
    ds = tfrecord.create_dataset([filename], 1, tf.float32)
    for value in ds:
        value = np.squeeze(value.numpy())
        assert value.shape == (tiff_image[0].shape[0], tiff_image[0].shape[1])
        assert np.allclose(value, tiff_image[0][:, :, 2])

def test_mix(tiff_image, tmp_path):
    filename1 = str(tmp_path / 'test1.tfrecord')
    filename2 = str(tmp_path / 'test2.tfrecord')
    image = tiff.TiffImage(tiff_image[1])
    tfrecord.image_to_tfrecord(image, [filename1, filename2], (30, 30), show_progress=False)
    ds = tfrecord.create_dataset([filename1, filename2], image.num_bands(), tf.float32, compressed=False)
    area = 0
    for value in ds:
        value = np.squeeze(value.numpy())
        assert value.shape[0] == 30 or value.shape[0] == 4
        assert value.shape[1] == 30 or value.shape[1] == 2
        area += value.shape[0] * value.shape[1]
    assert area == 64 * 32
