# DELTA

DELTA (Deep Earth Learning, Tools, and Analysis) is a framework for deep learning on satellite imagery.

It is currently under active development.

## Installation

1. Run ./scripts/setup.sh to install DELTA's dependencies in Ubuntu.
2. From the top directory, run `pip install --user -e .` to install for your user in
editable mode (linking to this directory). This will also install all dependencies with pip.

Note that DELTA requires python3 and tensorflow2.

## Tools

### Machine Learning Tools

`classify.py` - Classify and display a single image given a neural network.

`train_task_specific.py` - Train a classifier from a set of imagery and labels.

`convert_input_image_folder.py` - Convert input images into a .tfrecord format that other DELTA tools will read.  Converts all images in a folder to the output folder.  Sample call:

> `python bin/convert_input_image_folder.py --input-folder data/delta/worldview/  --output-folder data/delta/worldview_tfrecord  --image-type worldview  --num-processes 2  --mix-outputs`

`train_autoencoder.py` - Train an autoencoder network from tfrecord inputs.  All of the inputs to this are defined in a configuration file, see the sample config file for details.  Sample call:

> `python bin/train_autoencoder.py --config-file sample_config.txt`

### Data Fetching Tools

`fetch_hdds_images.py` - Use to programmatically fetch WorldView images from the USGS HDDS dataset.  Requires special (machine to machine) privileges from the website as well as an EarthExplorer login.


`get_landsat_dswe_labels.py`  - Use to programmatically fetch DSWE from the USGS that correspond to a given landsat file.  Requires special (machine to machine) privileges from the website as well as an EarthExplorer login.


`get_landsat_support_files.py` - Use to programmatically fetch SRTM files from the USGS that correspond to a given landsat file.  Requires special (machine to machine) privileges from the website as well as an EarthExplorer AND URS login.


# Example Use Cases

Train a classifier on a dataset.  First specify a config file, `your_config.yaml'

> `bin/delta train --config $PATH_TO_CONFIG/your_config.yaml name_of_model.h5`
