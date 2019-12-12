#pylint:disable=redefined-outer-name
import os
import random
import shutil
import tempfile
import zipfile

import numpy as np
import pytest

from delta.imagery.sources import tiff, tfrecord

def generate_tile(width=32, height=32, blocks=50):
    """Generate a widthXheightX3 image, with blocks pixels surrounded by ones and the rest zeros in band 0"""
    image = np.zeros((width, height, 1), np.float32)
    label = np.zeros((width, height), np.uint8)
    for _ in range(blocks):
        x = random.randint(2, width - 3)
        y = random.randint(2, height - 3)
        # ensure non-overlapping
        while (image[x + 2, y - 2 : y + 2, 0] > 0.0).any() or (image[x + 2, y - 2 : y + 2, 0] > 0.0).any() or \
              (image[x - 1 : x + 1, y - 2, 0] > 0.0).any() or (image[x - 1 : x + 1, y + 2, 0] > 0.0).any():
            x = random.randint(2, width - 3)
            y = random.randint(2, height - 3)
        image[x + 1, y - 1, 0] = 1.0
        image[x    , y - 1, 0] = 1.0
        image[x - 1, y - 1, 0] = 1.0
        image[x + 1, y    , 0] = 1.0
        image[x - 1, y    , 0] = 1.0
        image[x + 1, y + 1, 0] = 1.0
        image[x    , y + 1, 0] = 1.0
        image[x - 1, y + 1, 0] = 1.0
        label[x, y] = 1
    return (image, label)

@pytest.fixture(scope="session")
def original_file():
    tmpdir = tempfile.mkdtemp()
    image_path = os.path.join(tmpdir, 'image.tiff')
    label_path = os.path.join(tmpdir, 'label.tiff')

    (image, label) = generate_tile(64, 32, 50)
    tiff.write_tiff(image_path, image)
    tiff.write_tiff(label_path, label)
    yield (image_path, label_path)

    shutil.rmtree(tmpdir)

@pytest.fixture(scope="session")
def tfrecord_filenames(original_file):
    tmpdir = tempfile.mkdtemp()
    image_path = os.path.join(tmpdir, 'test.tfrecord')
    label_path = os.path.join(tmpdir, 'test.tfrecordlabel')
    tfrecord.image_to_tfrecord(tiff.TiffImage(original_file[0]), [image_path], tile_size=(30, 30))
    tfrecord.image_to_tfrecord(tiff.TiffImage(original_file[1]), [label_path], tile_size=(30, 30))
    #image_writer = tfrecord.make_tfrecord_writer(image_path)
    #label_writer = tfrecord.make_tfrecord_writer(label_path)
    #width = 32
    #height = 30
    #for i in range(1):
    #    for j in range(1):
    #        (image, label) = generate_tile(width, height)
    #        tfrecord.write_tfrecord_image(image, image_writer,
    #                                      i * width, j * height)
    #        tfrecord.write_tfrecord_image(label, label_writer,
    #                                      i * width, j * height)
    #image_writer.close()
    #label_writer.close()
    yield (image_path, label_path)

    shutil.rmtree(tmpdir)

@pytest.fixture(scope="session")
def worldview_filenames(original_file):
    tmpdir = tempfile.mkdtemp()
    image_name = 'WV02N42_939570W073_2520792013040400000000MS00_GU004003002'
    imd_name = '19MAY13164205-M2AS-503204071020_01_P003.IMD'
    zip_path = os.path.join(tmpdir, image_name + '.zip')
    label_path = os.path.join(tmpdir, image_name + '_label.tiff')
    image_dir = os.path.join(tmpdir, 'image')
    image_path = os.path.join(image_dir, image_name + '.tif')
    vendor_dir = os.path.join(image_dir, 'vendor_metadata')
    imd_path = os.path.join(vendor_dir, imd_name)
    os.mkdir(image_dir)
    os.mkdir(vendor_dir)
    # not really a valid file but this is all we need, only one band in image
    with open(imd_path, 'a') as f:
        f.write('absCalFactor = 9.295654e-03\n')
        f.write('effectiveBandwidth = 4.730000e-02\n')

    tiff.TiffImage(original_file[0]).save(image_path)
    tiff.TiffImage(original_file[1]).save(label_path)
    #(image, label) = generate_tile(width, height)
    #tiff.write_tiff(image_path, image)
    #tiff.write_tiff(label_path, label)

    z = zipfile.ZipFile(zip_path, mode='x')
    z.write(image_path, arcname=image_name + '.tif')
    z.write(imd_path, arcname=os.path.join('vendor_metadata', imd_name))
    z.close()

    yield (zip_path, label_path)

    shutil.rmtree(tmpdir)

NUM_SOURCES = 2
@pytest.fixture(scope="session")
def all_sources(tfrecord_filenames, worldview_filenames):
    return [(tfrecord_filenames, '.tfrecord', 'tfrecord', '.tfrecordlabel', 'tfrecord'),
            (worldview_filenames, '.zip', 'worldview', '_label.tiff', 'tiff')]
