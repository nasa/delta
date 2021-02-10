#!/bin/bash

# This example trains a Landsat 8 cloud classifier. This classification is
# based on the SPARCS validation data:
# https://www.usgs.gov/core-science-systems/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

if [ ! -f l8cloudmasks.zip ]; then
  echo "Downloading dataset."
  wget https://landsat.usgs.gov/cloud-validation/sparcs/l8cloudmasks.zip
fi

if [ ! -d sending ]; then
  echo "Extracting dataset."
  unzip -q l8cloudmasks.zip
  mkdir validate
  mv sending/LC82290562014157LGN00_24_data.tif sending/LC82210662014229LGN00_18_data.tif validate/
  mkdir train
  mv sending/*_data.tif train/
  mkdir labels
  mv sending/*_mask.png labels/
fi

if [ ! -f l8_clouds.h5 ]; then
  cp $SCRIPTPATH/l8_cloud.yaml .
  delta train --config l8_cloud.yaml l8_clouds.h5
fi

delta classify --config l8_cloud.yaml --image-dir ./validate --overlap 32 l8_clouds.h5
