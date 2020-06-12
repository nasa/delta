#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# use this repository and key for gdal-2.0
sudo add-apt-repository -y ppa:ubuntugis/ppa
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 089EBE08314DF160
sudo apt update

sudo apt install -y python3-dev python3-gdal || { echo >&2 "ERROR. Failed to install python3."; exit 1; }

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools requests numpy

$DIR/linter/install_linter.sh || { echo >&2 "ERROR. Failed to install linter."; exit 1; }

echo "All dependencies successfully installed."
