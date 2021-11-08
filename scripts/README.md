### Script Documentation

This folder contains various scripts and tools that have been created while developing DELTA

classify_directory.py - Tool to run Delta, FIST presoak, and HMTFIST on a set of input images

HMTFIST_caller.py - Tool to provide a convenient command-line interface to the HMTFIST tool

presoak_directory.py - Tool to run presoak on a batch of input images


## convert

landsat_toa.py - Script to apply Top of Atmosphere (ToA) correction to Landsat images

save_tiffs.py - Write out input images specified in a Delta config file as tiffs

wordview_toa.py - Script to apply Top of Atmosphere (ToA) correction to Worldview images

## docker

Contains setup files to create a Docker image for running Delta

## example

Location for storing sample execution scripts

## fetch

check_inputs.py - Verifies that all input images specified in a Delta config file can be properly loaded

fetch_hdds_images.py - Script to automate downloads of Worldview flood data from the HHDS web site

get_landsat_dswe_labels.py - Script to automatically download DSWE label images corresponding to a Landsat image

get_landsat_support_files.py - Script to automatically download SRTM images corresponding to a Landsat image

random_folder_split.py - Tool for creating folders of symlinks to a set of data in a way that partitions the files randomly into train/validate sets

unpack_inputs.py - Tool for batch unpacking of input image files

## linter

Contains the linter configuration files

## visualize

Contains some tools for comparing images