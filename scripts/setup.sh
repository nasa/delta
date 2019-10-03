#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# use this repository and key for gdal-2.0
sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 089EBE08314DF160
sudo apt update

sudo apt install -y python3-pip python3-dev python3-gdal || { echo >&2 "ERROR. Failed to install pip3."; exit 1; }
pip3 install --user -q pylint pytest || { echo >&2 "ERROR. Failed to install prequired packages."; exit 1; }

$DIR/linter/install_linter.sh || { echo >&2 "ERROR. Failed to install linter."; exit 1; }

echo "All dependencies successfully installed."
