DELTA (Deep Earth Learning, Tools, and Analysis) is a framework for deep learning on satellite imagery.

DELTA is currently under active development by the [NASA Ames Intelligent Robotics Group](https://ti.arc.nasa.gov/tech/asr/groups/intelligent-robotics/).

## Installation

1. Install python3 and GDAL. In Ubuntu, you can run `./scripts/setup.sh` to do this (GDAL in the Ubuntu package manager is / was broken).
2. From the top directory, containing `setup.py`, run `pip install --user -e .` to install for your user in
editable mode (linking to this directory). This will also install all dependencies with pip.

## Using DELTA

### Training

Train a neural network with::

```
  delta train [ --config config.yaml ... ] output.h5
```

Both images and corresponding labels must be specified. The trained network will attempt to map the input images
to the given labels. 

### Classification

Classify an image using a previously trained neural network with::

```
  delta classify [ --config config.yaml ... ] output.h5
```

For each input image, a tiff file with the extension `_predicted.tiff` will
be output. If labels are also provided, an error image showing incorrectly labeled
pixels and a confusion matrix pdf file will be generated as well.

### MLFlow

DELTA integrates with [MLFlow](http://mlflow.org) to track training. MLFlow options can
be specified in the corresponding area of the configuration file. By default, training and
validation metrics are logged, along with all configuration parameters. The most recent neural
network is saved to a file when the training program is interrupted or completes.

View all the logged training information through mlflow by running::

```
  delta mlflow_ui
```

and navigating to the printed URL in a browser.

## Configuration Files

DELTA is configured with YAML files. Some options can be overwritten with command line options (use `--help` to see which).

All available configuration options and their default values are shown [here](./delta/config/delta.yaml).
We suggest that users create one reusable configuration file to describe each dataset, and a separate configuration file to train
on or classify that dataset.

