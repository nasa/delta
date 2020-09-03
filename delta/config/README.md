DELTA Configuration Files
=========================
DELTA is configured with [YAML files](https://yaml.org/spec/1.2/spec.html). For an example with
all options, showing all parameters DELTA and their default values, see [delta.yaml](./delta.yaml).

`delta` accepts multiple config files on the command line. For example, run

    delta train --config dataset.yaml --config train.yaml

to train on a dataset specified by `dataset.yaml` with training parameters given in `train.yaml`.
Parameters can be overriden globally for all runs of `delta` as well, by placing options in
`$HOME/.config/delta/delta.yaml` on Linux. This is only recommended for global parameters
such as the cache directory.

Most options, aside from the input images and labels, have reasonable defaults. We recommend
only setting the necessary options.
Note that some configuration options can be overwritten on the command line: run
`delta --help` to see which.

The remainder of this document details the available configuration parameters. Note that
DELTA is still under active development and parts are likely to change in the future.

Dataset
-----------------
Images and labels are specified with the `images` and `labels` fields respectively,
within `dataset`. Both share the
same underlying options.

 * `type`: Indicates which loader to use, e.g., `tiff` for geotiff.
   The available loaders are listed [here](../imagery/sources/README.md).
 * Files to load must be specified in one of three ways:
   * `directory` and `extension`: Use all images in the directory ending with the given extension.
   * `file_list`: Provide a text file with one image file name per line.
   * `files`: Provide a list of file names in yaml.
 * `preprocess`: Supports limited image preprocessing. Currently only scaling is supported. We recommend
   scaling input imagery in the range 0.0 to 1.0 for best results with most of our networks.
   * `enabled`: Turn preprocessing on or off.
   * `scale_factor`: Factor to scale all readings by.
 * `nodata_value`: A pixel value to ignore in the images.

As an example:

  ```
  dataset:
    images:
      type: worldview
      directory: images/
    labels:
      type: tiff
      directory: labels/
      extension: _label.tiff
  ```

This configuration will load worldview files ending in `.zip` from the `images/` directory.
It will then find matching tiff files ending in `_label.tiff` from the `labels` directory
to use as labels.

Train
-----
These options are used in the `delta train` command.

 * `network`: The nueral network to train. See the next section for details.
 * `chunk_stride`: When collecting training samples, skip every `n` pixels between adjacent blocks. Keep the 
   default of 1 to use all available training data.
 * `batch_size`: The number of chunks to train on in a group. May affect convergence speed. Larger
   batches allow higher training data throughput, but may encounter memory limitations.
 * `steps`: If specified, stop training for each epoch after the given number of batches.
 * `epochs`: the number of times to iterate through all training data during training.
 * `loss_function`: [Keras loss function](https://keras.io/losses/). For integer classes, use
   `sparse_categorical_cross_entropy`.
 * `metrics`: A list of [Keras metrics](https://keras.io/metrics/) to evaluate.
 * `optimizer`: The [Keras optimizer](https://keras.io/optimizers/) to use.
 * `validation`: Specify validation data. The validation data is tested after each epoch to evaluate the
   classifier performance. Always use separate training and validation data!
   * `from_training` and `steps`: If `from_training` is true, take the `steps` training batches
     and do not use it for training but for validation instead.
   * `images` and `labels`: Specified using the same format as the input data. Use this imagery as testing data
     if `from_training` is false.

### Network

These options configure the neural network to train with the `delta train` command.

 * `classes`: The number of classes in the input data. The classes must currently have values
   0 - n in the label images.
 * `model`: The network structure specification.
   folder. You can either point to another `yaml_file`, such as the ones in the delta/config/networks
   directory, or specify one under the `model` field in the same format as these files. The network
   layers are specified using the [Keras functional layers API](https://keras.io/layers/core/)
   converted to YAML files.


MLFlow
------
Used in the `delta train` and `delta mlflow_ui` commands to keep track of training runs using MLFlow.

 * `enabled`: Turn MLFlow use off or on.
 * `uri`: The URI for where MLFlow should store tracking runs. Options such as file directories, databases,
   and HTTP servers are supported. See the
   [`mlflow.set_tracking_uri()`](https://www.mlflow.org/docs/latest/tracking.html)  documentation for details.
 * `experiment_name`: A name for the experiment to track in MLFlow.
 * `frequency`: Record metrics after this many batches. Want to pick a number that won't slow down training or
   use too much disk space.
 * `checkpoints`: Configure saving of checkpoint networks to mlflow, in case something goes wrong or to compare
   networks from different stages of training.
   * `frequency`: Frequency in batches to save a checkpoint. Networks can require a fair amount of disk space,
     so don't save too often.
   * `only_save_latest`: If true, only keep the network file from the most recent checkpoint.

TensorBoard
-----------
[Tensorboard](https://www.tensorflow.org/tensorboard) is TensorFlow's visualization toolkit.

 * `enabled`: Set to true to save data to tensorboard. Disabled by default.
 * `dir`: Specify a directory to save tensorboard data.

General
-------

 * `gpus`: The number of GPUs to use, or `-1` for all.
 * `threads`: The number of threads to use for loading images into tensorflow.
 * `block_size_mb`: The size of blocks in images to load at a time. If too small may be data starved.
 * `tile_ratio` The ratio of block width and height when loading images. Can affect disk use efficiency.
 * `cache`: Configure cacheing options. The subfield `dir` specifies a directory on disk to store cached files,
   and `limit` is the number of files to retain in the cache. Used mainly for image types
   which much be extracted from archive files.
