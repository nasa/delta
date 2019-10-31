"""
Functions to support images stored as TFRecord.
"""

from delta.imagery import tfrecord_utils

from . import basic_sources

class TFRecordImage(basic_sources.DeltaImage):
    def __init__(self, path, _, num_regions):
        super(TFRecordImage, self).__init__(num_regions)
        self.path = path
        self._num_bands = None
        self._size = None

    def prep(self):
        pass

    def read(self, roi=None):
        raise NotImplementedError()

    def __get_bands_size(self):
        self._num_bands, width, height = tfrecord_utils.get_record_info(self.path)
        self._size = (width, height)

    def num_bands(self):
        if self._num_bands is None:
            self.__get_bands_size()
        return self._num_bands

    def size(self):
        if self._size is None:
            self.__get_bands_size()
        return self._size
