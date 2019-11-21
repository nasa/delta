#pylint: disable=redefined-outer-name
from osgeo import gdal
import pytest
import numpy as np
import tensorflow as tf

from delta.imagery.sources import tfrecord, tiff

@pytest.fixture(scope="function")
def tiff_image(tmp_path):
    width = 64
    height = 32
    numpy_image = np.random.rand(width, height, 3)
    filename = str(tmp_path / 'test.tiff')
    tiff.write_tiff(filename, numpy_image, data_type=gdal.GDT_Float32)
    return (numpy_image, filename)

def test_save_tfrecord(tiff_image, tmp_path):
    filename = str(tmp_path / 'test.tfrecord')
    image = tiff.TiffImage(tiff_image[1])
    tfrecord.image_to_tfrecord(image, [filename], image.size(), show_progress=False)
    ds = tfrecord.create_dataset([filename], image.num_bands(), tf.float32)
    for value in ds.take(1):
        value = np.squeeze(value.numpy())
        assert value.shape == tiff_image[0].shape
        assert np.allclose(value, tiff_image[0])
