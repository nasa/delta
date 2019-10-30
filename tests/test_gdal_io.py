"""
Test for GDAL I/O classes.
"""
import pytest
import numpy as np

from delta.imagery import utilities

from delta.imagery import rectangle
from delta.imagery.image_reader import MultiTiffFileReader
from delta.imagery.image_writer import TiffWriter

def check_landsat_tiff(filename):
    '''
    Checks reading landsat tiffs.
    '''
    input_reader = MultiTiffFileReader()
    input_reader.load_images([filename])
    assert input_reader.image_size() == (37, 37)
    assert input_reader.num_bands() == 8
    for i in range(1, input_reader.num_bands() + 1):
        (bsize, (blocks_x, blocks_y)) = input_reader.get_block_info(i)
        assert bsize == [37, 6]
        assert blocks_x == 1
        assert blocks_y == 7
        assert utilities.gdal_dtype_to_numpy_type(input_reader.data_type(i)) == np.float32
        assert input_reader.nodata_value(i) is None

    meta = input_reader.get_all_metadata()
    geo = meta['geotransform']
    assert geo[0] == pytest.approx(-122.3, abs=0.01)
    assert geo[1] == pytest.approx(0.0, abs=0.01)
    assert geo[2] == pytest.approx(0.0, abs=0.01)
    assert geo[3] == pytest.approx(37.5, abs=0.01)
    assert geo[4] == pytest.approx(0.0, abs=0.01)
    assert geo[5] == pytest.approx(0.0, abs=0.01)
    assert 'gcps' in meta
    assert 'gcpproj' in meta
    assert 'projection' in meta
    assert 'metadata' in meta

    r = rectangle.Rectangle(0, 0, width=input_reader.image_size()[0],
                            height=input_reader.image_size()[0])
    d1 = input_reader.read_roi(r)
    assert d1.shape == (input_reader.image_size()[0], input_reader.image_size()[1],
                        input_reader.num_bands())

def check_same(filename1, filename2):
    '''
    Checks whether or not two files are the same
    '''
    in1 = MultiTiffFileReader()
    in2 = MultiTiffFileReader()
    in1.load_images([filename1])
    in2.load_images([filename2])
    assert in1.image_size() == in2.image_size()
    assert in1.num_bands() == in2.num_bands()
    for i in range(1, in1.num_bands() + 1):
        assert in1.get_block_info(i) == in2.get_block_info(i)
        assert in1.data_type(i) == in2.data_type(i)
        assert in1.nodata_value(i) == in2.nodata_value(i)

    m_1 = in1.get_all_metadata()
    m_2 = in2.get_all_metadata()
    assert m_1['geotransform'] == m_2['geotransform']
    assert m_1['gcps'] == m_2['gcps']
    assert m_1['gcpproj'] == m_2['gcpproj']
    assert m_1['projection'] == m_2['projection']
    assert m_1['metadata'] == m_2['metadata']

    (width, height) = in1.image_size()
    d1 = in1.read_roi(rectangle.Rectangle(0, 0, width=width, height=height))
    d2 = in2.read_roi(rectangle.Rectangle(0, 0, width=width, height=height))
    assert np.array_equal(d1, d2)

def test_geotiff_read():
    '''
    Tests reading a landsat geotiff.
    '''
    check_landsat_tiff('data/landsat.tiff')

def test_geotiff_write(tmpdir):
    '''
    Tests writing a landsat geotiff.
    '''
    input_reader = MultiTiffFileReader()
    input_reader.load_images(['data/landsat.tiff'])
    new_tiff = tmpdir / 'test.tiff'

    (block_size, (blocks_x, blocks_y)) = input_reader.get_block_info(1)
    (cols, rows) = input_reader.image_size()

    writer = TiffWriter()
    writer.init_output_geotiff(str(new_tiff), cols, rows, input_reader.nodata_value(1),
                               tile_width=block_size[0],
                               tile_height=block_size[1],
                               metadata=input_reader.get_all_metadata(),
                               num_bands=input_reader.num_bands(),
                               data_type=input_reader.data_type(1))

    input_bounds = rectangle.Rectangle(0, 0, width=cols, height=rows)
    output_rois = []
    for row in range(0, blocks_y):
        for col in range(0, blocks_x):

            # Get the ROI for the block, cropped to fit the image size.
            roi = rectangle.Rectangle(col * block_size[0], row * block_size[1],
                                      width=block_size[0], height=block_size[1])
            roi = roi.get_intersection(input_bounds)
            output_rois.append(roi)

    def callback_function(output_roi, read_roi, data_vec):
        """Callback function to write the first channel to the output file."""

        # Figure out the output block
        col = output_roi.min_x // block_size[0]
        row = output_roi.min_y // block_size[1]

        # Figure out where the desired output data falls in read_roi
        x_0 = output_roi.min_x - read_roi.min_x
        y_0 = output_roi.min_y - read_roi.min_y
        x_1 = x_0 + output_roi.width()
        y_1 = y_0 + output_roi.height()
        assert x_0 == 0
        assert y_0 == 0

        # Crop the desired data portion and write it out.
        for i in range(input_reader.num_bands()):
            output_data = data_vec[i][y_0:y_1, x_0:x_1]
            assert data_vec[i].shape == (y_1 - y_0, x_1 - x_0)
            writer.write_geotiff_block(output_data, col, row, band=i)

    input_reader.process_rois(output_rois, callback_function)
    writer.finish_writing_geotiff()
    writer.cleanup()

    check_same('data/landsat.tiff', str(new_tiff))
