DELTA Configuration Files
=========================
DELTA is configured with [YAML files](https://yaml.org/spec/1.2/spec.html). For an example with
all options, showing all parameters DELTA and their default values, see [delta.yaml](./delta.yaml).

`delta` accepts multiple config files on the command line. For example, run

```bash
delta train --config dataset.yaml --config train.yaml
```

to train on a dataset specified by `dataset.yaml`:

```yaml
dataset:
  images:
    type: tiff
    directory: train/
  labels:
    type: tiff
    directory: labels/
  classes: 2
```

with training parameters given in `train.yaml`:

```yaml
train:
  network:
    yaml_file: networks/convpool.yaml
  epochs: 10
```

Parameters can be overriden globally for all runs of `delta` as well, by placing options in
`$HOME/.config/delta/delta.yaml` on Linux. This is only recommended for global parameters
such as the cache directory.

Most options, aside from the input images and labels, have reasonable defaults. We recommend
only setting the necessary options.
Note that some configuration options can be overwritten on the command line: run
`delta --help` to see which.

The remainder of this document details the available configuration parameters.

Dataset
-----------------
Images and labels are specified with the `images` and `labels` fields respectively,
within `dataset`. Both share the
same underlying options.

 * `type`: Indicates which `delta.imagery.delta_image.DeltaImage` image reader to use, e.g., `tiff` for geotiff.
   The reader should previously be registered with `delta.config.extensions.register_image_reader`.
 * Files to load must be specified in one of three ways:
    * `directory` and `extension`: Use all images in the directory ending with the given extension.
    * `file_list`: Provide a text file with one image file name per line.
    * `files`: Provide a list of file names in yaml.
 * `preprocess`: Specify a preprocessing chain. We recommend
   scaling input imagery in the range 0.0 to 1.0 for best results with most of our networks.
   DELTA also supports custom preprocessing commands. Default actions include:
    * `scale` with `factor` argument: Divide all values by amount.
    * `offset` with `factor` argument: Add `factor` to pixel values.
    * `clip` with `bounds` argument: clip all pixels to bounds.
   Preprocessing commands are registered with `delta.config.extensions.register_preprocess`.
   A full list of defaults (and examples of how to create new ones) can be found in `delta.extensions.preprocess`.
 * `nodata_value`: A pixel value to ignore in the images. Will try to determine from the file if this is not specified.
 * `classes`: Either an integer number of classes or a list of individual classes. If individual classes are specified,
   each list item should be the pixel value of the class in the label images, and a dictionary with the
   following potential attributes (see example below):
    * `name`: Name of the class.
    * `color`: Integer to use as the RGB representation for some classification options.
    * `weight`: How much to weight the class during training (useful for underrepresented classes).

As an example:

```yaml
dataset:
  images:
    type: tiff
    directory: images/
    preprocess:
      - scale:
          factor: 256.0
    nodata_value: 0
  labels:
    type: tiff
    directory: labels/
    extension: _label.tiff
    nodata_value: 0
  classes:
    - 1:
        name: Cloud
        color: 0x0000FF
        weight: 2.0
    - 2:
        name: Not Cloud
        color: 0xFFFFFF
        weight: 1.0
```

This configuration will load tiff files ending in `.tiff` from the `images/` directory.
It will then find matching tiff files ending in `_label.tiff` from the `labels` directory
to use as labels. The image values will be divied by a factor of 256 before they are used.
(It is often helpful to scale images to a range of 0-1 before training.) The labels represent two classes:
clouds and non-clouds. Since there are fewer clouds, these are weighted more havily. The label
images should contain 0 for nodata, 1 for cloud pixels, and 2 for non-cloud pixels.

Train
-----
These options are used in the `delta train` command.

 * `network`: The nueral network to train. One of `yaml_file` or `layers` must be specified.
    * `yaml_file`: A path to a yaml file with only the params and layers fields. See `delta/config/networks`
      for examples.
    * `params`: A dictionary of parameters to substitute in the `layers` field.
    * `layers`: A list of layers which compose the network. See the following section for details.
 * `stride`: When collecting training samples, skip every `n` pixels between adjacent blocks. Keep the 
   default of ~ or 1 to use all available training data. Not used for fully convolutional networks.
 * `batch_size`: The number of patches to train on at a time. If running out of memory, reducing
   batch size may be helpful.
 * `steps`: If specified, stop training for each epoch after the given number of batches.
 * `epochs`: the number of times to iterate through all training data during training.
 * `loss`: [Keras loss function](https://keras.io/losses/). For integer classes, use
   `sparse_categorical_cross_entropy`. May be specified either as a string, or as a dictionary
   with arguments to pass to the loss function constructor. Custom losses registered with
   `delta.config.extensions.register_loss` may be used.
 * `metrics`: A list of [Keras metrics](https://keras.io/metrics/) to evaluate. Either the string
   name or a dictionary with the constructor arguments may be used. Custom metrics registered with
   `delta.config.extensions.register_metric` or loss functions may also be used.
 * `optimizer`: The [Keras optimizer](https://keras.io/optimizers/) to use. May be specified as a string or
   as a dictionary with constructor parameters.
 * `callbacks`: A list of [Keras callbacks)(https://keras.io/api/callbacks/) to use during training, specified as
   either a string or as a dictionary with constructor parameters. Custom callbacks registered with
   `delta.config.extensions.register_metric` may also be used.
 * `validation`: Specify validation data. The validation data is tested after each epoch to evaluate the
   classifier performance. Always use separate training and validation data!
   * `from_training` and `steps`: If `from_training` is true, take the `steps` training batches
     and do not use it for training but for validation instead. If `from_training` is false, `steps` is ignored.
   * `images` and `labels`: Specified using the same format as the input data. Use this imagery as testing data
     if `from_training` is false.

Classify
-----
These options are used in the `delta classify` command.

 * `regions`: A list of region names to look for in WKT files associated with images.
 * `wkt_dir`: Directory to look for WKT files in.  If not specified they are expected to be in the same folders as input images.
 * `results_file`: Write a copy of the output statistics to this file
 * `metrics`: Include either losses or metrics here as specified in the Train section.  Currently some metrics
   will throw an exception when used this way, more support for them is planned.

```Sample config entries:
classify:
  regions:
   - sample_region_name
   - another_region
  wkt_dir: /alternate/wkt/location/
  results_file: log_here.txt
  metrics:
    - SparseRecall: # Works
        label: No Water
        name: sparse_recall
        binary: true
    - MappedDice: # Works!
        mapping:
          Water: 1.0
          No Water: 0.0
          Maybe Water: 0.5
          Cloud: 0.0
        name: dice 
```

By default when classify is run with labels available for the input image, it will compute some statistics
across all of the images and also on a per-image basis. You can also provide a WKT formatted shape file for
each input image containing one or more polygons/multipolygons, each with one or more region names. For each
region name specified in the config file, all regions including this name will have their statistics jointly
computed. In addition, all regions without a name will have their statistics individually computed. WKT files
should have the same names as their associated image files but with the extension ".wkt.csv".  There is a sample WKT file, along with a picture of the described regions, [here](../../docs/sample.wkt.csv)

### Network

For the `layers` attribute, any [Keras Layer](https://keras.io/api/layers/) can
be used, including custom layers registered with `delta.config.extensions.register_layer`.  

Sub-fields of the layer are argument names and values which are passed to the layer's constructor.  

A special sub-field, `inputs`, is a list of the names of layers to pass as inputs to this layer.
If `inputs` is not specified, the previous layer is used by default. Layer names can be specified `name`.

```yaml
layers:
  Input:
    shape: [~, ~, num_bands]
    name: input
  Add:
    inputs: [input, input]
```

This simple example takes an input and adds it to itself.

Since this network takes inputs of variable size ((~, ~, `num_bands`) is the input shape) it is a **fully
convolutional network**. This means that during training and classification, it will be evaluated on entire
tiles rather than smaller chunks.

A few special parameters are available by default:

 * `num_bands`: The number of bands / channels in an image.
 * `num_classes`: The number of classes provided in dataset.classes.

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
 * `verbose`: Trigger verbose printing.
 * `extensions`: List of extensions to load. Add custom modules here and they will be loaded when
   delta starts.

I/O
-------
 * `threads`: The number of threads to use for loading images into tensorflow.
 * `tile_size`: The size of a tile to load into memory at a time. For fully convolutional networks, the
   entire tile will be processed at a time, for others it will be chunked.
 * `cache`: Options for a cache, which is used by a few image types (currently worldview and landsat).
    * `dir`: Directory to store the cache. `default` gives a reasonable OS-specific default.
    * `limit`: Maximum number of items to store in the cache before deleting old entries.
