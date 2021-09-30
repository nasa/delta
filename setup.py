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

import os
import os.path
import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

scripts = []
for n in os.listdir('bin'):
    name = os.path.join('bin', n)
    if os.path.isfile(name) and os.access(name, os.X_OK):
        scripts.append(name)

setuptools.setup(
    name="delta",
    version="0.4.0",
    author="NASA Ames",
    author_email="todo@todo",
    description="Deep learning for satellite imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nasa/delta",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: GIS",
        "Operating System :: OS Independent"
    ],
    install_requires=[
        'tensorflow>=2.1',
        'tensorflow_addons',
        'usgs<0.3',
        'scipy',
        'matplotlib',
        'mlflow',
        'portalocker',
        'appdirs',
        'gdal',
        'shapely',
        'pillow'
        #'numpy', # these are included by tensorflow with restrictions
        #'h5py'
    ],
    scripts=scripts,
    include_package_data = True,
    package_data = {'' : ['*.cfg']},
    python_requires='>=3.6',
)
