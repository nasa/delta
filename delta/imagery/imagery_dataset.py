"""
Tools for loading data into the TensorFlow Dataset class.
"""
import functools
import os
import os.path
import tempfile

import numpy as np
import tensorflow as tf

from delta.imagery import utilities
from delta.imagery.sources import basic_sources
from delta.imagery.sources import landsat
from delta.imagery.sources import worldview

IMAGE_CLASSES = {
        'landsat' : landsat.LandsatImage,
        'worldview' : worldview.WorldviewImage,
        'rgba' : basic_sources.RGBAImage
}

class ImageryDataset:
    # TODO: something better with num_regions, chunk_size
    # TODO: Need to clean up this whole class!
    """
    Create dataset with all files in image_folder with extension ext.

    Cache list of files to list_path, and use caching folder cache_folder.
    """
    def __init__(self, image_type, image_folder=None, chunk_size=256,
                 chunk_overlap=0, list_path=None):
        try:
            im_type = IMAGE_CLASSES[image_type]
        except IndexError:
            raise Exception('Unrecognized input type: ' + image_type)
        exts = im_type.extensions()
        self._num_bands = im_type.num_bands()

        if list_path is None:
            tempfd, list_path = tempfile.mkstemp()
            os.close(tempfd)
            self.__temp_path = list_path
        else:
            self.__temp_path = None

        self._image_type = image_type
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Generate a text file list of all the input images, plus region indices.
        self._num_regions, self._num_images = self.__make_image_list(image_folder, list_path, exts)
        assert self._num_regions > 0


        # This dataset returns the lines from the text file as entries.
        ds = tf.data.TextLineDataset(list_path)

        # This function generates fake label info for loaded data.
        label_gen_function = functools.partial(self.__load_fake_labels)

        def generate_chunks(lines):
            y = tf.py_function(self.__load_data, [lines], [tf.float64])
            y[0].set_shape((0, self._num_bands, chunk_size, chunk_size))
            return y

        def generate_labels(lines):
            y = tf.py_function(label_gen_function, [lines], [tf.int32])
            y[0].set_shape((0, 1))
            return y

        # Tell TF to use the functions above to load our data.
        chunk_set = ds.map(generate_chunks, num_parallel_calls=1)
        label_set = ds.map(generate_labels, num_parallel_calls=1)

        # Break up the chunk sets to individual chunks
        # TODO: Does this improve things?
        chunk_set = chunk_set.flat_map(tf.data.Dataset.from_tensor_slices)
        label_set = label_set.flat_map(tf.data.Dataset.from_tensor_slices)

        # Pair the data and labels in our dataset
        ds = tf.data.Dataset.zip((chunk_set, label_set))

        # Filter out all chunks with zero (nodata) values
        self._ds = ds.filter(lambda chunk, label: tf.math.equal(tf.math.zero_fraction(chunk), 0))

    def __del__(self):
        if self.__temp_path is not None:
            os.remove(self.__temp_path)

    def image_class(self):
        return IMAGE_CLASSES[self._image_type]

    def __load_data(self, text_line):
        text_line = text_line.numpy().decode() # Convert from TF to string type
        parts  = text_line.split(',')
        path   = parts[0].strip()
        roi = utilities.Rectangle(int(parts[1].strip()), int(parts[2].strip()),
                                  int(parts[3].strip()), int(parts[4].strip()))

        image = self.image_class()(path)
        return image.chunk_image_region(roi, self._chunk_size, self._chunk_overlap)

    # TODO: delete this and load actual labels
    def __load_fake_labels(self, text_line):
        """Use to generate fake label data for load_image_region"""
        text_line = text_line.numpy().decode() # Convert from TF to string type
        parts  = text_line.split(',')
        path   = parts[0].strip()
        roi = utilities.Rectangle(int(parts[1].strip()), int(parts[2].strip()),
                                  int(parts[3].strip()), int(parts[4].strip()))

        image = self.image_class()(path)
        chunk_data = image.chunk_image_region(roi, self._chunk_size, self._chunk_overlap)

        # Make a fake label
        full_shape = chunk_data.shape[0]
        chunk_data = np.zeros(full_shape, dtype=np.int32)
        chunk_data[ 0:10] = 1 # Junk labels
        chunk_data[10:20] = 2
        return chunk_data

    def __make_image_list(self, top_folder, output_path, exts):
        '''Write a file listing all of the files in a (recursive) folder
           matching the provided extension.
        '''

        num_entries = 0
        num_images = 0
        with open(output_path, 'w') as f:
            for root, dummy_directories, filenames in os.walk(top_folder):
                for filename in filenames:
                    if os.path.splitext(filename)[1] in exts:
                        path = os.path.join(root, filename)
                        rois = self.image_class()(path).tiles()
                        for r in rois:
                            f.write('%s,%d,%d,%d,%d\n' % (path, r.min_x, r.min_y, r.max_x, r.max_y))
                            num_entries += 1
                    num_images += 1
        return num_entries, num_images

    def dataset(self):
        return self._ds

    def num_images(self):
        return self._num_images
    def num_regions(self):
        return self._num_regions
    def num_bands(self):
        return self._num_bands
    def chunk_size(self):
        return self._chunk_size
