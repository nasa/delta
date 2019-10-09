#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# use this repository and key for gdal-2.0
sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 089EBE08314DF160
sudo apt update

sudo apt install -y python3-pip python3-dev python3-gdal || { echo >&2 "ERROR. Failed to install pip3."; exit 1; }

pip3 install --user -q pylint psutil usgs pytest || { echo >&2 "ERROR. Failed to install prequired packages."; exit 1; }
pip3 install --user -q numpy scipy matplotlib || { echo >&2 "ERROR. Failed to install one of numpy, scipy, matplotlib."; exit 1; }

pip3 install --user GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}') --global-option=build_ext --global-option="-I/usr/include/gdal"/

pip3 install --user portalocker

pip3 install --user -q tensorflow || { echo >&2 "ERROR. Failed to install tensorflow."; exit 1; }

# To install gpu tensorflow uncomment this line
# pip3 install --user -q tensorflow-gpu || { echo >&2 "ERROR. Failed to install tensorflow."; exit 1; }

pip3 install --user -q mlflow || { echo >&2 "ERROR. Failed to install mlflow."; exit 1; }
pip3 install --user -q portalocker || { echo >&2 "ERROR. Failed to install mlflow."; exit 1; }


#command -v pip3>/dev/null 2>&1 || { echo >&2 "ERROR. Please install pip3."; exit 1; }

$DIR/linter/install_linter.sh || { echo >&2 "ERROR. Failed to install linter."; exit 1; }

echo "All dependencies successfully installed."


echo "TODO: Get data from known location"
mkdir -p $DIR/../data/{in/toy_data,out/mlflow}

# ssh -fNL 12345:gramps.ndc.nasa.gov:22 $USER@wow.ndc.nasa.gov
# scp -P 12345 $USER@localhost://path/to/remote/file $DIR/../data/in/toy_data/ 
