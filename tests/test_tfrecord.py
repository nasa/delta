#pylint: disable=redefined-outer-name
from osgeo import gdal
import pytest
import numpy as np
import tensorflow as tf

from delta.imagery import utilities
from delta.imagery.sources import tfrecord, tiff

@pytest.fixture(scope="function")
def tiff_image(tmp_path):
    width = 64
    height = 32
    numpy_image = np.random.rand(width, height, 3)
    filename = str(tmp_path / 'test.tiff')
    image_writer = tiff.TiffWriter(filename, width, height, 3, data_type=gdal.GDT_Float32)
    for i in range(3):
        image_writer.write_block(numpy_image[:,:,i], 0, 0, i)
    del image_writer
    return (numpy_image, filename)

def test_save_tfrecord(tiff_image, tmp_path):
    filename = str(tmp_path / 'test.tfrecord')
    image = tiff.TiffImage(tiff_image[1])
    image_writer = tfrecord.make_tfrecord_writer(filename)
    tfrecord.write_tfrecord_image(image.read(), image_writer,
                                  0, 0, image.width(), image.height(), image.num_bands())
    image_writer.close()
    #tfrecord.image_to_tfrecord(image, [filename], image.size(), show_progress=False)
    raw_dataset = tf.data.TFRecordDataset(filenames=filename, compression_type=tfrecord.TFRECORD_COMPRESSION_TYPE)
    num_bands = image.num_bands()
    print(tf.float32, utilities.get_num_bytes_from_gdal_type(image.data_type()))
    raw_dataset = raw_dataset.map(lambda x: tfrecord.load_tensor(x, num_bands, tf.float32))
    for value in raw_dataset.take(1):
        value = np.squeeze(value.numpy())
        assert value.shape == tiff_image[0].shape
        assert np.allclose(value, tiff_image[0])
