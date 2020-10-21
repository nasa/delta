**DELTA** (Deep Earth Learning, Tools, and Analysis) is a framework for deep learning on satellite imagery,
based on Tensorflow. Use DELTA to train and run neural networks to classify large satellite images. DELTA
provides pre-trained autoencoders for a variety of satellites to reduce required training data
and time.

DELTA is currently under active development by the
[NASA Ames Intelligent Robotics Group](https://ti.arc.nasa.gov/tech/asr/groups/intelligent-robotics/). Expect
frequent changes. It is initially being used to map floods for disaster response, in collaboration with the
[U.S. Geological Survey](http://www.usgs.gov), [National Geospatial Intelligence Agency](https://www.nga.mil/),
[National Center for Supercomputing Applications](http://www.ncsa.illinois.edu/), and
[University of Alabama](https://www.ua.edu/). DELTA is a component of the
[Crisis Mapping Toolkit](https://github.com/nasa/CrisisMappingToolkit), in addition
to our previous software for mapping floods with Google Earth Engine.

Installation
============

1. Install [python3](https://www.python.org/downloads/), [GDAL](https://gdal.org/download.html), and the [GDAL python bindings](https://pypi.org/project/GDAL/).
   For Ubuntu Linux, you can run `scripts/setup.sh` from the DELTA repository to install these dependencies.

2. Install Tensorflow with pip following the [instructions](https://www.tensorflow.org/install). For
   GPU support in DELTA (highly recommended) follow the directions in the
   [GPU guide](https://www.tensorflow.org/install/gpu).

3. Checkout the delta repository and install with pip:

   ```
   git clone http://github.com/nasa/delta
   python3 -m pip install delta
   ```

  This installs DELTA and all dependencies (except for GDAL which must be installed manually in step 1).

Usage
=====

As a simple example, consider training a neural network to map water in Worldview imagery.
You would:

1. **Collect** training data. Find and save Worldview images with and without water. For a robust
   classifier, the training data should be as representative as possible of the evaluation data.

2. **Label** training data. Create images matching the training images pixel for pixel, where each pixel
   in the label is 0 if it is not water and 1 if it is.

3. **Train** the neural network. Run
   ```
   delta train --config wv_water.yaml wv_water.h5
   ```
   where `wv_water.yaml` is a configuration file specifying the labeled training data and any
   training parameters (learn more about configuration files below). The command will output a
   neural network file `wv_water.h5` which can be
   used for classification. The neural network operates on the level of *chunks*, inputting
   and output smaller blocks of the image at a time.

4. **Classify** with the trained network. Run
   ```
   delta classify --image image.tiff wv_water.h5
   ```
   to classify `image.tiff` using the network `wv_water.h5` learned previously.
   The file `image_predicted.tiff` will be written to the current directory showing the resulting labels.

Configuration Files
-------------------

DELTA is configured with YAML files. Some options can be overwritten with command line options (use
`delta --help` to see which). [Learn more about DELTA configuration files](./delta/config/README.md).

All available configuration options and their default values are shown [here](./delta/config/delta.yaml).
We suggest that users create one reusable configuration file to describe the parameters specific
to each dataset, and separate configuration files to train on or classify that dataset.

Supported Image Formats
-----------------------
DELTA supports tiff files and a few other formats.
Users can extend DELTA with their own custom formats. We are looking to expand DELTA to support other
useful file formats.

MLFlow
------

DELTA integrates with [MLFlow](http://mlflow.org) to track training. MLFlow options can
be specified in the corresponding area of the configuration file. By default, training and
validation metrics are logged, along with all configuration parameters. The most recent neural
network is saved to a file when the training program is interrupted or completes.

View all the logged training information through mlflow by running::

```
  delta mlflow_ui
```

and navigating to the printed URL in a browser. This makes it easier to keep track when running
experiments and adjusting parameters.

Using DELTA from Code
=====================
You can also call DELTA as a python library and customize it with your own extensions, for example,
custom image types. The python API documentation can be generated as HTML. To do so:

```
  pip install pdoc3
  ./scripts/docs.sh
```

Then open `html/delta/index.html` in a web browser.

Contributors
============
We welcome pull requests to contribute to DELTA. However, due to NASA legal restrictions, we must require
that all contributors sign and submit a
[NASA Individual Contributor License Agreement](https://www.nasa.gov/sites/default/files/atoms/files/astrobee_individual_contributor_license_agreement.pdf).
You can scan the document and submit via email. Thank you for your understanding.

Important notes for developers:

 * **Branching**: Active development occurs on `develop`. Releases are pushed to `master`.

 * **Code Style**: Code must pass our linter before merging. Run `scripts/linter/install_linter.sh` to install
   the linter as a git pre-commit hook.

 * **Unit Tests**: Code must pass unit tests before merging. Run `pytest` in the `tests` directory to run the tests.
   Please add new unit tests as appropriate.

 * **Development Setup**: You can install delta using pip's `-e` flag which installs in editable mode. Then you can
   run `delta` and it will use your latest changes made to the repo without reinstalling.

Licensing
=========
DELTA is released under the Apache 2 license.

Copyright (c) 2020, United States Government, as represented by the Administrator of the National Aeronautics and Space Administration. All rights reserved.
