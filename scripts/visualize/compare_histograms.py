#!/usr/bin/python3

# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script creates a pdf comparing the histograms of all tiff files input.

import sys

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

from delta.extensions.sources.tiff import TiffImage

def plot_band(names, band):
    imgs = [TiffImage(n) for n in names]
    max_value = 8.0
    for img in imgs:
        data = np.ndarray.flatten(img.read(bands=band))
        data = data[data != 0.0]
        data[data > max_value] = max_value
        plt.hist(data, bins=200, alpha=0.5)
        plt.title('Band ' + str(band))
    pdf.savefig()
    plt.close()

assert len(sys.argv) > 1, 'No  input tiffs specified.'

with PdfPages('output.pdf') as pdf:
    for i in range(8):
        plot_band(sys.argv[1:], i)
